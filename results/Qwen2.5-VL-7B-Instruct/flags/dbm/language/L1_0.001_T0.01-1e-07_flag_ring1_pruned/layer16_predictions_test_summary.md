# layer16_predictions_test (entity=flags)

## target_attribute=language

overall accuracy: 57.8% (n=13332)
final_score = 1/2(cause + mean(iso)) = 1/2(60.4% + 56.9%) = **58.6%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 54.1% | 21.0% | 54.1% |
| capital (iso) | 3600 | 62.0% | 31.8% | 62.0% |
| currency (iso) | 3096 | 54.6% | 23.9% | 54.6% |
| language (cause) | 3036 | 60.4% | 60.4% | 35.8% |
