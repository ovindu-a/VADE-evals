# DBM layer sweep -- entity=flags attribute=language

positions=flag_ring1 l1_coef=0.001 temperature=0.01->1e-07 pruned=True eval_split=test

**best layer: 16 (final_score=58.6%)**

| layer | status | cause | iso_mean | final_score | n | dims selected |
|---|---|---|---|---|---|---|
| 10 | ok | 52.4% | 44.9% | 48.6% | 13332 | 2095/3584 (eps=0.01) |
| 16 | ok | 60.4% | 56.9% | 58.6% | 13332 | 1916/3584 (eps=0.01) |
| 20 | ok | 51.9% | 56.8% | 54.3% | 13332 | 1934/3584 (eps=0.01) |
| 22 | ok | 58.5% | 57.1% | 57.8% | 13332 | 2008/3584 (eps=0.01) |
