# predictions (entity=flags)

## target_attribute=currency

overall accuracy: 37.4% (n=4000)
final_score = 1/2(cause + mean(iso)) = 1/2(33.5% + 40.4%) = **36.9%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 1059 | 0.4% | 94.1% | 0.4% |
| capital (iso) | 988 | 40.8% | 41.7% | 40.8% |
| currency (cause) | 1015 | 33.5% | 33.5% | 18.6% |
| language (iso) | 938 | 79.9% | 14.4% | 79.9% |
