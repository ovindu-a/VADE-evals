# layer6_predictions_test (entity=flags)

## target_attribute=language

overall accuracy: 41.8% (n=13332)
final_score = 1/2(cause + mean(iso)) = 1/2(50.6% + 39.1%) = **44.9%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 36.3% | 52.3% | 36.3% |
| capital (iso) | 3600 | 44.2% | 52.7% | 44.2% |
| currency (iso) | 3096 | 36.8% | 40.8% | 36.8% |
| language (cause) | 3036 | 50.6% | 50.6% | 45.0% |
