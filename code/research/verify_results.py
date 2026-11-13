"""Check all published families, immutable caches and article effect directions."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    evidence = ROOT / 'evidence'
    claims = pd.read_csv(evidence / 'article_claims.csv')
    means = pd.read_csv(evidence / 'primary_metrics.csv')
    bank = pd.read_csv(evidence / 'bank_contrasts.csv')
    clinc = pd.read_csv(evidence / 'clinc_mc10000.csv')
    clinc_primary = pd.read_csv(evidence / 'clinc_primary.csv')
    multi = pd.read_csv(evidence / 'multiclass_contrasts.csv')
    assert [len(bank), len(clinc), len(clinc_primary), len(multi)] == [5691, 1698, 1698, 3564]
    provenance = json.loads((ROOT / 'provenance.json').read_text())['files']
    repository = ROOT.parents[1]
    checked = 0
    for name, entry in provenance.items():
        if name.startswith('code/research/'):
            assert sha(repository / name) == entry['sha256'], name
            checked += 1

    def point(dataset, pair, criterion, method, arm, metric='accuracy'):
        subset = means[(means.dataset == dataset) & (means.pair_id == pair) & (means.criterion == criterion)
                       & (means.method == method) & (means.arm == arm)]
        assert len(subset) == 1
        return 100 * subset.iloc[0][metric + '_mean']

    for row in claims.to_dict('records'):
        assert row['low'] <= row['difference_pp'] <= row['high']
        assert (row['low'] > 0 or row['high'] < 0) and row['adjusted_p'] < .05
        if row['family'] == 'B':
            value = sum(sign * (point('banking77', row['pair_id'], 'mse', row['method'], arm)
                                - point('banking77', row['pair_id'], 'mse', row['reference'], arm))
                        for arm, sign in [('harm8_raw', 1), ('natural_raw', -1)])
            table = bank
        elif row['family'] == 'C':
            value = sum(weight * point('clinc150', row['pair_id'], criterion, method, arm)
                        for arm, criterion, method, weight in json.loads(row['terms']))
            table = clinc
            assert row['primary_significant'] and row['mc_significant']
        else:
            value = point('clinc150', row['pair_id'], row['criterion'], row['method'], 'natural_raw', row['metric'])
            value -= point('clinc150', row['pair_id'], row['criterion'], row['reference'], 'natural_raw', row['metric'])
            table = multi[multi.dataset == 'clinc150']
        np.testing.assert_allclose(value, row['difference_pp'], atol=1e-10, rtol=0)
        source = table[(table.pair_id == row['pair_id']) & (table.method == row['source_method'])
                       & (table.reference == row['source_reference'])]
        for column in ['kind', 'arm', 'criterion', 'metric']:
            if column in table and pd.notna(row.get(column)):
                source = source[source[column] == row[column]]
        assert len(source) == 1, row['key']
        source = source.iloc[0]
        sign = 1 if source.method == row['method'] else -1
        expected = [sign * source.difference_pp,
                    source.low if sign == 1 else -source.high,
                    source.high if sign == 1 else -source.low, source.adjusted_p]
        np.testing.assert_allclose(expected, [row[k] for k in ['difference_pp', 'low', 'high', 'adjusted_p']], atol=1e-12)

    inputs = [
        (ROOT / 'data/imbalance_cache.npz', json.loads((ROOT / 'data/imbalance_manifest.json').read_text())['cache_sha256']),
        (ROOT / 'clinc150/artifacts/router_cache.npz', json.loads((ROOT / 'clinc150/artifacts/complete.json').read_text())['cache_sha256']),
    ]
    manifest = json.loads((ROOT / 'multiclass/labels/manifest.json').read_text())
    for dataset in ['banking77', 'clinc150']:
        inputs.append((ROOT / 'multiclass/labels' / f'{dataset}.npz', manifest[dataset]['labels_sha256']))
    for path, expected in inputs:
        assert sha(path) == expected
    gain = clinc[(clinc.kind == 'gain_interaction') & (clinc.criterion == 'accuracy') & clinc.primary_significant]
    assert len(gain) == 8 and gain.mc_significant.all()
    print(json.dumps({'status': 'passed', 'article_effects': len(claims), 'source_files': checked,
                      'input_caches': len(inputs), 'families': [len(bank), len(clinc), len(multi)],
                      'stable_gain_interactions': len(gain)}, indent=2))


if __name__ == '__main__':
    main()
