# heads_common10 (entity=flags)

mean final_score across 3 target attributes: **45.0%**

## target_attribute=calling_code

overall accuracy: 36.0% (n=9888)
final_score = 1/2(cause + mean(iso)) = 1/2(95.5% + 2.1%) = **48.8%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (cause) | 3600 | 95.5% | 95.5% | 0.2% |
| capital (iso) | 3600 | 0.1% | 98.0% | 0.1% |
| currency (iso) | 2688 | 4.2% | 72.1% | 4.2% |

## target_attribute=capital

overall accuracy: 36.9% (n=9888)
final_score = 1/2(cause + mean(iso)) = 1/2(98.0% + 2.2%) = **50.1%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 0.2% | 95.5% | 0.2% |
| capital (cause) | 3600 | 98.0% | 98.0% | 0.1% |
| currency (iso) | 2688 | 4.2% | 72.1% | 4.2% |

## target_attribute=currency

overall accuracy: 19.7% (n=9888)
final_score = 1/2(cause + mean(iso)) = 1/2(72.1% + 0.2%) = **36.1%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 0.2% | 95.5% | 0.2% |
| capital (iso) | 3600 | 0.1% | 98.0% | 0.1% |
| currency (cause) | 2688 | 72.1% | 72.1% | 4.2% |

## target_attribute=language

overall accuracy: 1.3% (n=9888)

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 0.2% | 95.5% | 0.2% |
| capital (iso) | 3600 | 0.1% | 98.0% | 0.1% |
| currency (iso) | 2688 | 4.2% | 72.1% | 4.2% |
