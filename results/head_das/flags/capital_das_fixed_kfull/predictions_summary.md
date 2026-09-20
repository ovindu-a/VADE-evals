# predictions (entity=flags)

## target_attribute=capital

overall accuracy: 26.6% (n=2000)
final_score = 1/2(cause + mean(iso)) = 1/2(97.9% + 2.2%) = **50.1%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 525 | 0.2% | 95.6% | 0.2% |
| capital (cause) | 512 | 97.9% | 97.9% | 0.6% |
| currency (iso) | 493 | 3.7% | 72.0% | 3.7% |
| language (iso) | 470 | 2.8% | 93.0% | 2.8% |
