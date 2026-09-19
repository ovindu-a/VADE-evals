# random_heads_common5 (entity=flags)

mean final_score across 2 target attributes: **49.6%**

## target_attribute=calling_code

overall accuracy: 29.0% (n=5072)
final_score = 1/2(cause + mean(iso)) = 1/2(0.0% + 100.0%) = **50.0%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (cause) | 3600 | 0.0% | 0.0% | 98.6% |
| capital (iso) | 1472 | 100.0% | 0.0% | 100.0% |

## target_attribute=capital

overall accuracy: 70.0% (n=5072)
final_score = 1/2(cause + mean(iso)) = 1/2(0.0% + 98.6%) = **49.3%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 98.6% | 0.0% | 98.6% |
| capital (cause) | 1472 | 0.0% | 0.0% | 100.0% |

## target_attribute=currency

overall accuracy: 99.0% (n=5072)

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 98.6% | 0.0% | 98.6% |
| capital (iso) | 1472 | 100.0% | 0.0% | 100.0% |

## target_attribute=language

overall accuracy: 99.0% (n=5072)

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 98.6% | 0.0% | 98.6% |
| capital (iso) | 1472 | 100.0% | 0.0% | 100.0% |
