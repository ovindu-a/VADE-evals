# heads_top8 (entity=flags)

mean final_score across 3 target attributes: **43.6%**

## target_attribute=calling_code

overall accuracy: 33.4% (n=9888)
final_score = 1/2(cause + mean(iso)) = 1/2(87.4% + 2.9%) = **45.1%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (cause) | 3600 | 87.4% | 87.4% | 1.2% |
| capital (iso) | 3600 | 0.3% | 96.5% | 0.3% |
| currency (iso) | 2688 | 5.4% | 70.7% | 5.4% |

## target_attribute=capital

overall accuracy: 37.1% (n=9888)
final_score = 1/2(cause + mean(iso)) = 1/2(96.5% + 3.3%) = **49.9%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 1.2% | 87.4% | 1.2% |
| capital (cause) | 3600 | 96.5% | 96.5% | 0.3% |
| currency (iso) | 2688 | 5.4% | 70.7% | 5.4% |

## target_attribute=currency

overall accuracy: 19.8% (n=9888)
final_score = 1/2(cause + mean(iso)) = 1/2(70.7% + 0.8%) = **35.7%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 1.2% | 87.4% | 1.2% |
| capital (iso) | 3600 | 0.3% | 96.5% | 0.3% |
| currency (cause) | 2688 | 70.7% | 70.7% | 5.4% |

## target_attribute=language

overall accuracy: 2.0% (n=9888)

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 1.2% | 87.4% | 1.2% |
| capital (iso) | 3600 | 0.3% | 96.5% | 0.3% |
| currency (iso) | 2688 | 5.4% | 70.7% | 5.4% |
