# layer16_predictions_test (entity=flags)

## target_attribute=language

overall accuracy: 60.7% (n=13332)
final_score = 1/2(cause + mean(iso)) = 1/2(60.8% + 60.5%) = **60.6%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 59.9% | 16.1% | 59.9% |
| capital (iso) | 3600 | 64.8% | 29.4% | 64.8% |
| currency (iso) | 3096 | 56.8% | 23.1% | 56.8% |
| language (cause) | 3036 | 60.8% | 60.8% | 35.4% |
