# full_image (entity=celebrities)

## target_attribute=birth_year

overall accuracy: 30.8% (n=1994)
final_score = 1/2(cause + mean(iso)) = 1/2(31.8% + 32.3%) = **32.1%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| birth_year (cause) | 531 | 31.8% | 31.8% | 1.9% |
| nationality (iso) | 519 | 5.8% | 69.6% | 5.8% |
| native_language (iso) | 489 | 0.2% | 85.9% | 0.2% |
| occupation (iso) | 455 | 91.0% | 5.1% | 91.0% |
