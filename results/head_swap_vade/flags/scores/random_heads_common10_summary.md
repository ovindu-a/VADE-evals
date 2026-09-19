# random_heads_common10 (entity=flags)

mean final_score across 3 target attributes: **47.0%**

## target_attribute=calling_code

overall accuracy: 59.0% (n=9888)
final_score = 1/2(cause + mean(iso)) = 1/2(0.0% + 91.5%) = **45.8%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (cause) | 3600 | 0.0% | 0.0% | 99.2% |
| capital (iso) | 3600 | 100.0% | 0.0% | 100.0% |
| currency (iso) | 2688 | 83.0% | 0.0% | 83.0% |

## target_attribute=capital

overall accuracy: 58.7% (n=9888)
final_score = 1/2(cause + mean(iso)) = 1/2(0.0% + 91.1%) = **45.5%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 99.2% | 0.0% | 99.2% |
| capital (cause) | 3600 | 0.0% | 0.0% | 100.0% |
| currency (iso) | 2688 | 83.0% | 0.0% | 83.0% |

## target_attribute=currency

overall accuracy: 72.5% (n=9888)
final_score = 1/2(cause + mean(iso)) = 1/2(0.0% + 99.6%) = **49.8%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 99.2% | 0.0% | 99.2% |
| capital (iso) | 3600 | 100.0% | 0.0% | 100.0% |
| currency (cause) | 2688 | 0.0% | 0.0% | 83.0% |

## target_attribute=language

overall accuracy: 95.1% (n=9888)

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 99.2% | 0.0% | 99.2% |
| capital (iso) | 3600 | 100.0% | 0.0% | 100.0% |
| currency (iso) | 2688 | 83.0% | 0.0% | 83.0% |
