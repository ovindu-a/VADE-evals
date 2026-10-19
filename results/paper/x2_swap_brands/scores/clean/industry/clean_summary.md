# clean (entity=brands)

## target_attribute=industry

overall accuracy: 64.2% (n=723)
final_score = 1/2(cause + mean(iso)) = 1/2(5.7% + 98.0%) = **51.9%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| founded_year (iso) | 231 | 96.1% | 0.0% | 96.1% |
| hq_country (iso) | 227 | 100.0% | 0.0% | 100.0% |
| industry (cause) | 265 | 5.7% | 5.7% | 83.0% |
