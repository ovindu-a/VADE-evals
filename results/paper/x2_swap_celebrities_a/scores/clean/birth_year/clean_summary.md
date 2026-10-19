# clean (entity=celebrities)

## target_attribute=birth_year

overall accuracy: 73.2% (n=1994)
final_score = 1/2(cause + mean(iso)) = 1/2(0.0% + 99.7%) = **49.9%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| birth_year (cause) | 531 | 0.0% | 0.0% | 68.7% |
| nationality (iso) | 519 | 99.2% | 0.0% | 99.2% |
| native_language (iso) | 489 | 100.0% | 0.0% | 100.0% |
| occupation (iso) | 455 | 100.0% | 0.0% | 100.0% |
