# heads_common10 (entity=flags)

mean final_score across 2 target attributes: **48.6%**

## target_attribute=calling_code

overall accuracy: 67.8% (n=5072)
final_score = 1/2(cause + mean(iso)) = 1/2(95.5% + 0.0%) = **47.8%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (cause) | 3600 | 95.5% | 95.5% | 0.2% |
| capital (iso) | 1472 | 0.0% | 98.6% | 0.0% |

## target_attribute=capital

overall accuracy: 28.7% (n=5072)
final_score = 1/2(cause + mean(iso)) = 1/2(98.6% + 0.2%) = **49.4%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 0.2% | 95.5% | 0.2% |
| capital (cause) | 1472 | 98.6% | 98.6% | 0.0% |

## target_attribute=currency

overall accuracy: 0.1% (n=5072)

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 0.2% | 95.5% | 0.2% |
| capital (iso) | 1472 | 0.0% | 98.6% | 0.0% |

## target_attribute=language

overall accuracy: 0.1% (n=5072)

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 0.2% | 95.5% | 0.2% |
| capital (iso) | 1472 | 0.0% | 98.6% | 0.0% |
