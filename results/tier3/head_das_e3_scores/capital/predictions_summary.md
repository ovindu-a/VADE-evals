# predictions (entity=flags)

## target_attribute=capital

overall accuracy: 55.4% (n=4000)
final_score = 1/2(cause + mean(iso)) = 1/2(68.0% + 52.6%) = **60.3%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 1056 | 13.3% | 13.8% | 13.3% |
| capital (cause) | 1025 | 68.0% | 68.0% | 16.5% |
| currency (iso) | 980 | 50.4% | 10.3% | 50.4% |
| language (iso) | 939 | 94.0% | 2.6% | 94.0% |
