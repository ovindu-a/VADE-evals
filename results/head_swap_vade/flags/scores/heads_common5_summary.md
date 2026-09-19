# heads_common5 (entity=flags)

mean final_score across 3 target attributes: **39.0%**

## target_attribute=calling_code

overall accuracy: 30.5% (n=9888)
final_score = 1/2(cause + mean(iso)) = 1/2(69.4% + 9.2%) = **39.3%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (cause) | 3600 | 69.4% | 69.4% | 5.3% |
| capital (iso) | 3600 | 2.4% | 92.0% | 2.4% |
| currency (iso) | 2688 | 16.0% | 49.1% | 16.0% |

## target_attribute=capital

overall accuracy: 39.8% (n=9888)
final_score = 1/2(cause + mean(iso)) = 1/2(92.0% + 10.7%) = **51.3%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 5.3% | 69.4% | 5.3% |
| capital (cause) | 3600 | 92.0% | 92.0% | 2.4% |
| currency (iso) | 2688 | 16.0% | 49.1% | 16.0% |

## target_attribute=currency

overall accuracy: 16.1% (n=9888)
final_score = 1/2(cause + mean(iso)) = 1/2(49.1% + 3.8%) = **26.5%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 5.3% | 69.4% | 5.3% |
| capital (iso) | 3600 | 2.4% | 92.0% | 2.4% |
| currency (cause) | 2688 | 49.1% | 49.1% | 16.0% |

## target_attribute=language

overall accuracy: 7.1% (n=9888)

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 5.3% | 69.4% | 5.3% |
| capital (iso) | 3600 | 2.4% | 92.0% | 2.4% |
| currency (iso) | 2688 | 16.0% | 49.1% | 16.0% |
