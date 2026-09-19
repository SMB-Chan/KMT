"""Distill validated Phi demonstrations into a small, offline numpy pilot."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from numbers import Real
from pathlib import Path
import time
import numpy as np
from policy import MLP
from ollama_pilot import Control, PilotError

FEATURES = ('altitude_m', 'forward_speed_m_s', 'vertical_speed_m_s',
            'wave_elevation_m', 'wave_rate_m_s', 'keel_clearance_m', 'water_mass_kg')
SCALES = np.array([25, 15, 5, 2, 3, 25, .5, 1, 25, 15], dtype=np.float64)


def features(observation, mission):
    if observation.get('failed') or mission.get('scenario') not in ('takeoff', 'landing'):
        raise PilotError('invalid or failed demonstration state')
    try:
        raw = [observation[k] for k in FEATURES] + [
            float(mission['scenario'] == 'landing'), mission['target_altitude_m'],
            mission['target_speed_m_s']]
        if any(not isinstance(v, Real) or isinstance(v, (bool, np.bool_)) for v in raw):
            raise ValueError('non-numeric feature')
        x = np.asarray(raw, dtype=np.float64) / SCALES
        if not np.isfinite(x).all():
            raise ValueError('nonfinite feature')
        return x
    except (KeyError, ValueError, TypeError) as exc:
        raise PilotError(f'invalid student observation: {exc}') from exc


class StudentPilot:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.body = MLP([10, 32, 32], rng)
        self.W = rng.normal(0, .1, (32, 2))
        self.b = np.zeros(2)

    def predict(self, x):
        h, _ = self.body.forward(np.atleast_2d(x))
        return np.tanh(h[-1] @ self.W + self.b)

    def fit(self, x, y, epochs=1000, lr=.03):
        if epochs <= 0 or not np.isfinite(lr) or lr <= 0:
            raise ValueError('epochs and learning rate must be positive')
        losses = []
        for _ in range(epochs):
            h, z = self.body.forward(x)
            pred = np.tanh(h[-1] @ self.W + self.b)
            losses.append(float(np.mean((pred-y)**2)))
            g = 2 * (pred-y) * (1-pred**2) / pred.size
            dW, db = h[-1].T @ g, g.sum(axis=0)
            body_grad = self.body.backward(h, z, g @ self.W.T)
            self.W -= lr * dW; self.b -= lr * db
            for layer, grad in zip(self.body.params, body_grad):
                layer['W'] -= lr * grad['W']; layer['b'] -= lr * grad['b']
        return losses

    def save(self, path):
        with Path(path).open('xb') as f:
            np.savez(f, version=np.array(1), body=self.body.get_flat(), W=self.W, b=self.b)

    @classmethod
    def load(cls, path):
        pilot = cls()
        with np.load(path, allow_pickle=False) as z:
            if (int(z['version']) != 1 or z['body'].shape != (pilot.body.flat_dim(),)
                    or z['W'].shape != (32, 2) or z['b'].shape != (2,)
                    or not all(np.isfinite(z[k]).all() for k in ('body', 'W', 'b'))):
                raise ValueError('invalid student checkpoint')
            pilot.body.set_flat(z['body'])
            pilot.W, pilot.b = z['W'].copy(), z['b'].copy()
        return pilot

    def check_model(self):
        return {'name':'phi-student', 'backend':'numpy', 'teacher':'phi3.5',
                'feature_version':1}

    def decide(self, observation, mission, previous=None):
        start = time.monotonic()
        raw = self.predict(features(observation, mission))[0]
        control = Control(float((raw[0]+1)/2), float(3.5+11.5*raw[1]))
        return control, {'latency_s':time.monotonic()-start, 'backend':'numpy'}


def load_demonstrations(paths):
    states, labels, provenance = [], [], []
    skipped = 0
    for path in paths:
        path = Path(path)
        content = path.read_bytes()
        accepted = 0
        for number, line in enumerate(content.decode().splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            # Never learn scripted fallbacks, student outputs or aborted requests.
            if record.get('source') != 'ollama':
                skipped += 1; continue
            try:
                control = Control.parse(json.dumps(record['control']))
                state = features(record['observation'], record['mission'])
            except (KeyError, PilotError) as exc:
                raise ValueError(f'{path}:{number}: invalid teacher record') from exc
            states.append(state)
            labels.append([2*control.throttle-1, (control.pitch_deg-3.5)/11.5])
            accepted += 1
        provenance.append({'path':str(path.resolve()), 'sha256':hashlib.sha256(content).hexdigest(),
                           'accepted':accepted})
    if not states:
        raise ValueError('no usable Ollama demonstrations')
    return np.array(states), np.array(labels), provenance, skipped


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('logs', nargs='+', type=Path, help='teacher decisions.jsonl files')
    p.add_argument('--output', type=Path, required=True, help='new directory')
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    x, y, sources, skipped = load_demonstrations(args.logs)
    student = StudentPilot(args.seed)
    initial = float(np.mean((student.predict(x)-y)**2))
    student.fit(x, y, args.epochs)
    final = float(np.mean((student.predict(x)-y)**2))
    args.output.mkdir(parents=True, exist_ok=False)
    student.save(args.output/'student.npz')
    report = dict(samples=len(x), skipped=skipped, seed=args.seed, epochs=args.epochs,
                  initial_training_mse=initial, final_training_mse=final, sources=sources,
                  evaluation='training fit only; no held-out flight performance established')
    (args.output/'training.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
