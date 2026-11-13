"""Interpret full BANKING77 contrasts; extracted from the original analysis stage."""
import argparse
from pathlib import Path
import pandas as pd


def interpret(contrasts):
    t = contrasts.copy()
    t['significant'] = (t.low > 0) | (t.high < 0)
    counts = t.groupby('kind').agg(total=('significant', 'size'), significant=('significant', 'sum'))
    gain = t[t.gain_only].groupby('kind').agg(gain_total=('significant', 'size'), gain_significant=('significant', 'sum'))
    counts = counts.join(gain).fillna(0).astype(int)
    simple = t[t.kind == 'simple'].set_index(['pair_id', 'arm', 'method', 'reference'])
    interactions = []
    for row in t[t.kind == 'interaction'].itertuples():
        natural = simple.loc[(row.pair_id, 'natural_raw', row.method, row.reference)]
        enriched = simple.loc[(row.pair_id, row.arm, row.method, row.reference)]
        record = row._asdict()
        record.update(natural_difference_pp=natural.difference_pp, natural_low=natural.low, natural_high=natural.high,
                      enriched_difference_pp=enriched.difference_pp, enriched_low=enriched.low, enriched_high=enriched.high,
                      point_ranking_flip=natural.difference_pp * enriched.difference_pp < -1e-12,
                      supported_crossover=((natural.low > 0 and enriched.high < 0) or (natural.high < 0 and enriched.low > 0)))
        interactions.append(record)
    interactions = pd.DataFrame(interactions).drop(columns='Index')
    profiles = []
    prof = t[t.kind == 'profile'].set_index(['pair_id', 'method', 'reference', 'metric'])
    for (pair, arm, method, reference), row in simple.iterrows():
        if arm != 'natural_raw':
            continue
        rescue = prof.loc[(pair, method, reference, 'rescue_precision')]
        harm = prof.loc[(pair, method, reference, 'harm_rate')]
        profiles.append({'pair_id': pair, 'method': method, 'reference': reference, 'gain_only': bool(row.gain_only),
                         'accuracy_difference_pp': row.difference_pp, 'accuracy_low': row.low, 'accuracy_high': row.high,
                         'accuracy_significant': bool(row.significant),
                         'rescue_precision_difference_pp': rescue.difference_pp, 'rescue_low': rescue.low, 'rescue_high': rescue.high,
                         'rescue_significant': bool(rescue.significant),
                         'harm_rate_difference_pp': harm.difference_pp, 'harm_low': harm.low, 'harm_high': harm.high,
                         'harm_significant': bool(harm.significant),
                         'profile_without_accuracy_detection': not row.significant and (rescue.significant or harm.significant)})
    return counts, interactions, pd.DataFrame(profiles)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--contrasts', type=Path, default=Path(__file__).parent / 'evidence/bank_contrasts.csv')
    args = parser.parse_args()
    counts, interactions, profiles = interpret(pd.read_csv(args.contrasts))
    print(counts.to_string())
    print('Supported crossovers:', int(interactions.supported_crossover.sum()))
