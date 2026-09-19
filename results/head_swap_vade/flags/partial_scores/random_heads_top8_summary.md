# random_heads_top8 (entity=flags)

mean final_score across 2 target attributes: **50.0%**

## target_attribute=calling_code

overall accuracy: 29.0% (n=5072)
final_score = 1/2(cause + mean(iso)) = 1/2(0.0% + 100.0%) = **50.0%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (cause) | 3600 | 0.0% | 0.0% | 100.0% |
| capital (iso) | 1472 | 100.0% | 0.0% | 100.0% |

## target_attribute=capital

overall accuracy: 71.0% (n=5072)
final_score = 1/2(cause + mean(iso)) = 1/2(0.0% + 100.0%) = **50.0%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 100.0% | 0.0% | 100.0% |
| capital (cause) | 1472 | 0.0% | 0.0% | 100.0% |

## target_attribute=currency

overall accuracy: 100.0% (n=5072)

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 100.0% | 0.0% | 100.0% |
| capital (iso) | 1472 | 100.0% | 0.0% | 100.0% |

## target_attribute=language

overall accuracy: 100.0% (n=5072)

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 100.0% | 0.0% | 100.0% |
| capital (iso) | 1472 | 100.0% | 0.0% | 100.0% |
