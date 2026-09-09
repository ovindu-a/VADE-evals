# layer10_predictions_test (entity=flags)

## target_attribute=language

overall accuracy: 46.7% (n=13332)
final_score = 1/2(cause + mean(iso)) = 1/2(52.4% + 44.9%) = **48.6%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 41.1% | 41.6% | 41.1% |
| capital (iso) | 3600 | 51.4% | 44.4% | 51.4% |
| currency (iso) | 3096 | 42.1% | 37.9% | 42.1% |
| language (cause) | 3036 | 52.4% | 52.4% | 44.1% |
