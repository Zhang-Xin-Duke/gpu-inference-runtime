"""Matched one/multiple-replica sweeps using the independent-process generator.

Workers must already be running. This script never provisions or stops hosts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endpoints', nargs='+', required=True)
    p.add_argument('--replica-counts', nargs='+', type=int, default=[1, 2])
    p.add_argument('--rates', nargs='+', type=float, default=[100, 500, 1000, 2000])
    p.add_argument('--seeds', nargs='+', type=int, default=[7, 17, 27])
    p.add_argument('--duration', type=float, default=30)
    p.add_argument('--deadline-ms', type=float, default=1000)
    p.add_argument('--rpc-concurrency', type=int, default=32)
    p.add_argument('--max-requests', type=int, default=256)
    p.add_argument('--kind', choices=['dense', 'sequence'], default='dense')
    p.add_argument('--features', type=int, default=256)
    p.add_argument('--hidden', type=int, default=512)
    p.add_argument('--embedding', type=int, default=64)
    p.add_argument('--samples', type=int, default=1)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    if len(set(a.endpoints)) != len(a.endpoints):
        p.error('endpoints must be distinct')
    if any(len(set(values)) != len(values) for values in (a.replica_counts,a.rates,a.seeds)):
        p.error('replica counts, rates and seeds must each be unique')
    if any(n < 1 or n > len(a.endpoints) for n in a.replica_counts):
        p.error('each replica count must fit the supplied endpoint list')
    if a.duration <= 0 or any(r <= 0 for r in a.rates):
        p.error('duration and rates must be positive')
    if not 0 < a.deadline_ms <= 10000 or not 1 <= a.rpc_concurrency <= 32:
        p.error('invalid deadline or RPC concurrency')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    plan = []
    for seed in a.seeds:
        for rate in a.rates:
            # Reverse order on alternate seeds to reduce a fixed ordering bias.
            counts = a.replica_counts if a.seeds.index(seed) % 2 == 0 else list(reversed(a.replica_counts))
            for count in counts:
                name = f'replicas-{count}-rps-{rate:g}-seed-{seed}'
                output = a.output_dir / (name + '.json')
                cmd = [sys.executable, str(Path(__file__).with_name('extended.py')),
                       '--endpoints', *a.endpoints[:count], '--duration', str(a.duration),
                       '--rps', str(rate), '--seed', str(seed), '--deadline-ms', str(a.deadline_ms),
                       '--rpc-concurrency', str(a.rpc_concurrency), '--max-requests', str(a.max_requests),
                       '--kind', a.kind, '--features', str(a.features), '--hidden', str(a.hidden),
                       '--embedding', str(a.embedding), '--samples', str(a.samples), '--output', str(output)]
                plan.append(dict(name=name, command=cmd))
    (a.output_dir / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    for cell in plan:
        print('START', cell['name'], flush=True)
        subprocess.run(cell['command'], check=True, timeout=a.duration + 240)
    files = sorted(a.output_dir.glob('*.json'))
    (a.output_dir / 'SHA256SUMS').write_text(''.join(
        hashlib.sha256(f.read_bytes()).hexdigest() + '  ' + f.name + '\n' for f in files))


if __name__ == '__main__':
    main()
