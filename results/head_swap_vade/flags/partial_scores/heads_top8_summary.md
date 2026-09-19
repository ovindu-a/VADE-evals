# heads_top8 (entity=flags)

mean final_score across 2 target attributes: **46.4%**

## target_attribute=calling_code

overall accuracy: 62.1% (n=5072)
final_score = 1/2(cause + mean(iso)) = 1/2(87.4% + 0.1%) = **43.8%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (cause) | 3600 | 87.4% | 87.4% | 1.2% |
| capital (iso) | 1472 | 0.1% | 96.7% | 0.1% |

## target_attribute=capital

overall accuracy: 28.9% (n=5072)
final_score = 1/2(cause + mean(iso)) = 1/2(96.7% + 1.2%) = **49.0%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 1.2% | 87.4% | 1.2% |
| capital (cause) | 1472 | 96.7% | 96.7% | 0.1% |

## target_attribute=currency

overall accuracy: 0.9% (n=5072)

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 1.2% | 87.4% | 1.2% |
| capital (iso) | 1472 | 0.1% | 96.7% | 0.1% |

## target_attribute=language

overall accuracy: 0.9% (n=5072)

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 1.2% | 87.4% | 1.2% |
| capital (iso) | 1472 | 0.1% | 96.7% | 0.1% |
