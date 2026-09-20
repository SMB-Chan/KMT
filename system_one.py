"""Local System One primitives: typed questions, probabilities, confidence."""
from dataclasses import dataclass
import math


def softmax(logits):
    peak = max(logits)
    weights = [math.exp(v - peak) for v in logits]
    total = sum(weights)
    return [w / total for w in weights]


def margin_confidence(probabilities):
    ordered = sorted(probabilities, reverse=True)
    if len(ordered) < 2:
        return float(ordered[0])
    return float(max(0.0, min(1.0, ordered[0] - ordered[1])))


@dataclass(frozen=True)
class Choice:
    key: str
    options: tuple


@dataclass(frozen=True)
class Score:
    key: str


@dataclass(frozen=True)
class Noul:
    key: str


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict
    confidence: float


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    confidence: float


@dataclass(frozen=True)
class NoulAnswer:
    noul: float
    confidence: float


def choice_from_logits(options, logits):
    if len(options) != len(logits) or len(options) < 2:
        raise ValueError('choice needs matching logits for at least two options')
    probs = softmax(logits)
    table = {name: float(p) for name, p in zip(options, probs)}
    return ChoiceAnswer(max(table, key=table.get), table, margin_confidence(probs))


def score_answer(value, confidence):
    score = float(max(0.0, min(1.0, value)))
    conf = float(max(0.0, min(1.0, confidence)))
    return ScoreAnswer(score, conf)


def noul_answer(probability):
    p = float(max(0.0, min(1.0, probability)))
    return NoulAnswer(p, float(max(p, 1.0 - p)))


def sigmoid(value):
    value = max(-20.0, min(20.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


class Logistic:
    """Binary calibrator: P(y=1|x) = sigmoid(w·x)."""

    def __init__(self, weights):
        import numpy as np
        self.weights = np.asarray(weights, dtype=float).reshape(-1)

    def predict(self, features):
        import numpy as np
        x = np.asarray(features, dtype=float)
        if x.ndim == 1:
            if x.size != self.weights.size:
                raise ValueError('feature size mismatch')
            return sigmoid(float(self.weights @ x))
        return np.array([sigmoid(float(self.weights @ row)) for row in x])

    @classmethod
    def fit(cls, features, labels, steps=400, lr=0.3):
        import numpy as np
        x = np.asarray(features, dtype=float)
        y = np.asarray(labels, dtype=float).reshape(-1)
        if x.ndim != 2 or len(x) != len(y) or len(x) == 0:
            raise ValueError('features must be a nonempty 2-D array')
        weights = np.zeros(x.shape[1])
        for _ in range(steps):
            p = np.array([sigmoid(float(weights @ row)) for row in x])
            weights -= lr * (x.T @ (p - y)) / len(y)
        return cls(weights)

    def save(self, path):
        import numpy as np
        np.savez(path, weights=self.weights)

    @classmethod
    def load(cls, path):
        import numpy as np
        return cls(np.load(path)['weights'])
