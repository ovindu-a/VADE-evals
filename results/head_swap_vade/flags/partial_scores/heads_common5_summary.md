# heads_common5 (entity=flags)

mean final_score across 2 target attributes: **42.4%**

## target_attribute=calling_code

overall accuracy: 50.2% (n=5072)
final_score = 1/2(cause + mean(iso)) = 1/2(69.4% + 3.2%) = **36.3%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (cause) | 3600 | 69.4% | 69.4% | 5.3% |
| capital (iso) | 1472 | 3.2% | 91.8% | 3.2% |

## target_attribute=capital

overall accuracy: 30.4% (n=5072)
final_score = 1/2(cause + mean(iso)) = 1/2(91.8% + 5.3%) = **48.5%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 5.3% | 69.4% | 5.3% |
| capital (cause) | 1472 | 91.8% | 91.8% | 3.2% |

## target_attribute=currency

overall accuracy: 4.7% (n=5072)

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 5.3% | 69.4% | 5.3% |
| capital (iso) | 1472 | 3.2% | 91.8% | 3.2% |

## target_attribute=language

overall accuracy: 4.7% (n=5072)

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 5.3% | 69.4% | 5.3% |
| capital (iso) | 1472 | 3.2% | 91.8% | 3.2% |
