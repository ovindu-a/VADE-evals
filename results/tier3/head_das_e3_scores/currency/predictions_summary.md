# predictions (entity=flags)

## target_attribute=currency

overall accuracy: 63.3% (n=4000)
final_score = 1/2(cause + mean(iso)) = 1/2(49.4% + 69.4%) = **59.4%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 1059 | 27.0% | 10.4% | 27.0% |
| capital (iso) | 988 | 88.6% | 2.2% | 88.6% |
| currency (cause) | 1015 | 49.4% | 49.4% | 10.6% |
| language (iso) | 938 | 92.6% | 2.0% | 92.6% |
