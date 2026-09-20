# predictions (entity=flags)

## target_attribute=currency

overall accuracy: 20.0% (n=2000)
final_score = 1/2(cause + mean(iso)) = 1/2(76.1% + 1.0%) = **38.5%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 526 | 0.6% | 94.5% | 0.6% |
| capital (iso) | 497 | 0.2% | 98.4% | 0.2% |
| currency (cause) | 507 | 76.1% | 76.1% | 4.3% |
| language (iso) | 470 | 2.1% | 93.2% | 2.1% |
