# clean (entity=celebrities)

## target_attribute=occupation

overall accuracy: 67.8% (n=1505)
final_score = 1/2(cause + mean(iso)) = 1/2(0.0% + 97.3%) = **48.6%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| birth_year (iso) | 365 | 92.9% | 0.0% | 92.9% |
| nationality (iso) | 376 | 98.9% | 0.0% | 98.9% |
| native_language (iso) | 309 | 100.0% | 0.0% | 100.0% |
| occupation (cause) | 455 | 0.0% | 0.0% | 100.0% |
