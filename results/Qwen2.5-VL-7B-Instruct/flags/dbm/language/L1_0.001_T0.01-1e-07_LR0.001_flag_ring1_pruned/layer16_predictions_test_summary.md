# layer16_predictions_test (entity=flags)

## target_attribute=language

overall accuracy: 56.2% (n=13332)
final_score = 1/2(cause + mean(iso)) = 1/2(50.3% + 57.7%) = **54.0%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 56.8% | 18.4% | 56.8% |
| capital (iso) | 3600 | 62.5% | 31.3% | 62.5% |
| currency (iso) | 3096 | 53.9% | 27.0% | 53.9% |
| language (cause) | 3036 | 50.3% | 50.3% | 45.9% |
