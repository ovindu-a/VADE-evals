# layer16_predictions_test (entity=flags)

## target_attribute=language

overall accuracy: 60.1% (n=13332)
final_score = 1/2(cause + mean(iso)) = 1/2(62.1% + 59.3%) = **60.7%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 58.4% | 17.4% | 58.4% |
| capital (iso) | 3600 | 63.9% | 30.6% | 63.9% |
| currency (iso) | 3096 | 55.7% | 24.4% | 55.7% |
| language (cause) | 3036 | 62.1% | 62.1% | 34.0% |
