# heads_heldout_per_attr (entity=flags)

## target_attribute=currency

overall accuracy: 19.5% (n=14052)
final_score = 1/2(cause + mean(iso)) = 1/2(74.4% + 0.9%) = **37.6%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 0.1% | 95.6% | 0.1% |
| capital (iso) | 3600 | 0.2% | 97.8% | 0.2% |
| currency (cause) | 3564 | 74.4% | 74.4% | 4.5% |
| language (iso) | 3288 | 2.3% | 93.9% | 2.3% |
