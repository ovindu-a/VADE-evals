# predictions (entity=flags)

## target_attribute=language

overall accuracy: 68.9% (n=4000)
final_score = 1/2(cause + mean(iso)) = 1/2(14.0% + 85.7%) = **49.9%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 1057 | 87.9% | 1.0% | 87.9% |
| capital (iso) | 997 | 99.2% | 0.1% | 99.2% |
| currency (iso) | 1010 | 70.0% | 1.0% | 70.0% |
| language (cause) | 936 | 14.0% | 14.0% | 73.0% |
