#!/usr/bin/env python3
"""Replay the frozen matched-feature statistics using aggregate returns only.

This adapter changes input packaging, not the estimator. Public benchmark returns
must be obtained separately from their provider. No stock observations, weights,
or trained checkpoints are loaded, and no model is fitted.
"""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
import numpy as np
import pyarrow.parquet as pq
from evaluate_paper_a_matched_control import sha, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'pricing-targets', 'training-dir', 'output-dir']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    c = json.loads(a.config.read_text())
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True)
    inputs = a.output_dir / 'aggregate_inputs'
    projected = a.output_dir / 'aggregate_training_projection'
    inputs.mkdir(); projected.mkdir()
    table = pq.read_table(a.pricing_targets)
    feature_months = table['month'].to_numpy(zero_copy_only=False).astype('datetime64[M]').astype(int)
    asset_columns = [x for x in table.column_names if x.startswith('asset_')]
    assert len(asset_columns) == 74 and len(np.unique(feature_months)) == len(feature_months)
    assets = np.column_stack([table[x].to_numpy(zero_copy_only=False).astype(np.float32) for x in asset_columns])
    for split in ['validation', 'development']:
        start, end = c['splits'][split]
        expected = np.arange(np.datetime64(start, 'M'), np.datetime64(end, 'M') + 1).astype(int)
        indexes = np.searchsorted(feature_months, expected - 1)
        np.testing.assert_array_equal(feature_months[indexes] + 1, expected)
        assert np.isfinite(assets[indexes]).all()
        np.savez_compressed(inputs / (split + '.npz'), assets=assets[indexes], target_months=expected)
    write_json(inputs / 'output_manifest.json', {
        'scope': 'Public benchmark returns and calendar only; no stock panel',
        'source_sha256': sha(a.pricing_targets), 'config_sha256': sha(a.config),
        'outputs': [{'path': f.name, 'sha256': sha(f)} for f in sorted(inputs.glob('*.npz'))]})
    original = json.loads((a.training_dir / 'output_manifest.json').read_text())
    original_hashes = {x['path']: x['sha256'] for x in original['outputs']}
    for arm in c['arms']:
        for k in c['factor_counts']:
            for seed in c['seeds']:
                folder = f'{arm}-k{k}-seed{seed}'
                for name in ['monthly_factor_returns.csv', 'quality_report.json']:
                    relative = folder + '/' + name
                    source = a.training_dir / relative
                    # The numerical return files are identical in the public package.
                    # JSON deployment paths can be sanitized; source/release hashes
                    # for those files are recorded in the release SOURCE_MANIFEST.
                    if name.endswith('.csv') and sha(source) != original_hashes[relative]:
                        raise RuntimeError('Aggregate return hash mismatch: ' + relative)
                    destination = projected / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, destination)
    write_json(projected / 'output_manifest.json', {
        'scope': 'Explicit public aggregate projection, not a replacement of the original training manifest',
        'original_training_manifest_sha256': sha(a.training_dir / 'output_manifest.json'),
        'outputs': [{'path': str(f.relative_to(projected)), 'sha256': sha(f)}
                    for f in sorted(projected.rglob('*')) if f.is_file()]})
    evaluator = Path(__file__).with_name('evaluate_paper_a_matched_control.py')
    subprocess.run([sys.executable, str(evaluator), '--config', str(a.config),
                    '--data-dir', str(inputs), '--training-dir', str(projected),
                    '--output-dir', str(a.output_dir / 'evaluation')], check=True)


if __name__ == '__main__':
    main()
