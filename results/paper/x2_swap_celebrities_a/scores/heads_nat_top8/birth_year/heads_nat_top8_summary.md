# heads_nat_top8 (entity=celebrities)

## target_attribute=birth_year

overall accuracy: 41.0% (n=1994)
final_score = 1/2(cause + mean(iso)) = 1/2(3.8% + 56.0%) = **29.9%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| birth_year (cause) | 531 | 3.8% | 3.8% | 20.3% |
| nationality (iso) | 519 | 31.2% | 51.3% | 31.2% |
| native_language (iso) | 489 | 37.2% | 53.6% | 37.2% |
| occupation (iso) | 455 | 99.6% | 0.4% | 99.6% |
