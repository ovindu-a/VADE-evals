# predictions (entity=flags)

## target_attribute=calling_code

overall accuracy: 58.3% (n=4000)
final_score = 1/2(cause + mean(iso)) = 1/2(93.0% + 46.9%) = **70.0%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (cause) | 1025 | 93.0% | 93.0% | 0.7% |
| capital (iso) | 995 | 40.5% | 42.0% | 40.5% |
| currency (iso) | 1015 | 20.7% | 32.5% | 20.7% |
| language (iso) | 965 | 79.5% | 15.5% | 79.5% |
