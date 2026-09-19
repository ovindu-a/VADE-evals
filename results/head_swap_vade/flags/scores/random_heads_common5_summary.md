# random_heads_common5 (entity=flags)

mean final_score across 3 target attributes: **46.8%**

## target_attribute=calling_code

overall accuracy: 58.9% (n=9888)
final_score = 1/2(cause + mean(iso)) = 1/2(0.0% + 91.3%) = **45.6%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (cause) | 3600 | 0.0% | 0.0% | 98.6% |
| capital (iso) | 3600 | 100.0% | 0.0% | 100.0% |
| currency (iso) | 2688 | 82.6% | 0.0% | 82.6% |

## target_attribute=capital

overall accuracy: 58.3% (n=9888)
final_score = 1/2(cause + mean(iso)) = 1/2(0.0% + 90.6%) = **45.3%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 98.6% | 0.0% | 98.6% |
| capital (cause) | 3600 | 0.0% | 0.0% | 100.0% |
| currency (iso) | 2688 | 82.6% | 0.0% | 82.6% |

## target_attribute=currency

overall accuracy: 72.3% (n=9888)
final_score = 1/2(cause + mean(iso)) = 1/2(0.0% + 99.3%) = **49.6%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 98.6% | 0.0% | 98.6% |
| capital (iso) | 3600 | 100.0% | 0.0% | 100.0% |
| currency (cause) | 2688 | 0.0% | 0.0% | 82.6% |

## target_attribute=language

overall accuracy: 94.8% (n=9888)

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 98.6% | 0.0% | 98.6% |
| capital (iso) | 3600 | 100.0% | 0.0% | 100.0% |
| currency (iso) | 2688 | 82.6% | 0.0% | 82.6% |
