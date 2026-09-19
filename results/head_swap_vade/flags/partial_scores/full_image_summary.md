# full_image (entity=flags)

mean final_score across 2 target attributes: **48.6%**

## target_attribute=calling_code

overall accuracy: 68.9% (n=5072)
final_score = 1/2(cause + mean(iso)) = 1/2(97.1% + 0.0%) = **48.5%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (cause) | 3600 | 97.1% | 97.1% | 0.0% |
| capital (iso) | 1472 | 0.0% | 97.6% | 0.0% |

## target_attribute=capital

overall accuracy: 28.3% (n=5072)
final_score = 1/2(cause + mean(iso)) = 1/2(97.6% + 0.0%) = **48.8%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 0.0% | 97.1% | 0.0% |
| capital (cause) | 1472 | 97.6% | 97.6% | 0.0% |

## target_attribute=currency

overall accuracy: 0.0% (n=5072)

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 0.0% | 97.1% | 0.0% |
| capital (iso) | 1472 | 0.0% | 97.6% | 0.0% |

## target_attribute=language

overall accuracy: 0.0% (n=5072)

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 0.0% | 97.1% | 0.0% |
| capital (iso) | 1472 | 0.0% | 97.6% | 0.0% |
