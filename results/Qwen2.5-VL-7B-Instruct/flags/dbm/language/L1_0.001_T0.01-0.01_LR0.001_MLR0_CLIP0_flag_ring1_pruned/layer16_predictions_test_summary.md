# layer16_predictions_test (entity=flags)

## target_attribute=language

overall accuracy: 59.8% (n=13332)
final_score = 1/2(cause + mean(iso)) = 1/2(61.0% + 59.3%) = **60.2%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 58.3% | 17.1% | 58.3% |
| capital (iso) | 3600 | 64.0% | 30.3% | 64.0% |
| currency (iso) | 3096 | 55.6% | 24.1% | 55.6% |
| language (cause) | 3036 | 61.0% | 61.0% | 35.0% |
