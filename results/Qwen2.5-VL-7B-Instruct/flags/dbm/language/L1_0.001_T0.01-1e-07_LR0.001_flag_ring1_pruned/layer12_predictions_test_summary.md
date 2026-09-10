# layer12_predictions_test (entity=flags)

## target_attribute=language

overall accuracy: 47.9% (n=13332)
final_score = 1/2(cause + mean(iso)) = 1/2(62.3% + 43.6%) = **53.0%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 37.4% | 42.2% | 37.4% |
| capital (iso) | 3600 | 51.5% | 43.8% | 51.5% |
| currency (iso) | 3096 | 41.9% | 36.6% | 41.9% |
| language (cause) | 3036 | 62.3% | 62.3% | 34.3% |
