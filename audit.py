"""Reproducible local audit; writes a machine-readable report without training artifacts."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/audit.json')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    checks = []
    with tempfile.TemporaryDirectory(prefix='hama-audit-') as cache:
        env = dict(os.environ, MPLCONFIGDIR=cache)
        for name, command in [('regressions', ['-m', 'unittest', 'discover', '-s', 'tests', '-v']),
                              ('physics', ['validate.py'])]:
            result = subprocess.run([sys.executable, *command], cwd=root, env=env,
                                    capture_output=True, text=True, timeout=120)
            checks.append(dict(name=name, passed=result.returncode == 0,
                               output=result.stdout + result.stderr))
            print(f"{name}: {'PASS' if result.returncode == 0 else 'FAIL'}")
    report = dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  python=sys.version, checks=checks,
                  source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted([*root.glob('*.py'), *root.glob('tests/*.py')])})
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(f'Report: {target.resolve()}')
    return 0 if all(c['passed'] for c in checks) else 1


if __name__ == '__main__':
    sys.exit(main())
