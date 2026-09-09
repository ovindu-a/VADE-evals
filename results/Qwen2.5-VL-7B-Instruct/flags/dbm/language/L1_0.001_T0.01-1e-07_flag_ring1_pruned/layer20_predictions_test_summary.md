# layer20_predictions_test (entity=flags)

## target_attribute=language

overall accuracy: 56.0% (n=13332)
final_score = 1/2(cause + mean(iso)) = 1/2(51.9% + 56.8%) = **54.3%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 59.6% | 16.2% | 59.6% |
| capital (iso) | 3600 | 62.2% | 30.8% | 62.2% |
| currency (iso) | 3096 | 48.5% | 30.2% | 48.5% |
| language (cause) | 3036 | 51.9% | 51.9% | 43.6% |
