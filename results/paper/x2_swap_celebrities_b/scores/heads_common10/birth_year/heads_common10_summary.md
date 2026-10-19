# heads_common10 (entity=celebrities)

## target_attribute=birth_year

overall accuracy: 31.8% (n=1994)
final_score = 1/2(cause + mean(iso)) = 1/2(8.3% + 42.2%) = **25.2%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| birth_year (cause) | 531 | 8.3% | 8.3% | 12.4% |
| nationality (iso) | 519 | 16.0% | 64.4% | 16.0% |
| native_language (iso) | 489 | 11.0% | 77.3% | 11.0% |
| occupation (iso) | 455 | 99.6% | 0.4% | 99.6% |
