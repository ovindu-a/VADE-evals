# clean (entity=celebrities)

## target_attribute=nationality

overall accuracy: 71.8% (n=1863)
final_score = 1/2(cause + mean(iso)) = 1/2(1.8% + 97.2%) = **49.5%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| birth_year (iso) | 477 | 91.6% | 0.0% | 91.6% |
| nationality (cause) | 494 | 1.8% | 1.8% | 76.3% |
| native_language (iso) | 501 | 100.0% | 0.0% | 100.0% |
| occupation (iso) | 391 | 100.0% | 0.0% | 100.0% |
