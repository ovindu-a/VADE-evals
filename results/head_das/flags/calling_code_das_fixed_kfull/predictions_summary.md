# predictions (entity=flags)

## target_attribute=calling_code

overall accuracy: 26.1% (n=2000)
final_score = 1/2(cause + mean(iso)) = 1/2(95.1% + 2.3%) = **48.7%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (cause) | 512 | 95.1% | 95.1% | 0.2% |
| capital (iso) | 497 | 0.2% | 98.6% | 0.2% |
| currency (iso) | 509 | 4.9% | 71.7% | 4.9% |
| language (iso) | 482 | 1.9% | 92.7% | 1.9% |
