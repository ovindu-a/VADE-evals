# full_image (entity=flags)

mean final_score across 3 target attributes: **46.5%**

## target_attribute=calling_code

overall accuracy: 35.5% (n=9888)
final_score = 1/2(cause + mean(iso)) = 1/2(97.1% + 0.2%) = **48.6%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (cause) | 3600 | 97.1% | 97.1% | 0.0% |
| capital (iso) | 3600 | 0.2% | 97.3% | 0.2% |
| currency (iso) | 2688 | 0.2% | 84.4% | 0.2% |

## target_attribute=capital

overall accuracy: 35.5% (n=9888)
final_score = 1/2(cause + mean(iso)) = 1/2(97.3% + 0.1%) = **48.7%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 0.0% | 97.1% | 0.0% |
| capital (cause) | 3600 | 97.3% | 97.3% | 0.2% |
| currency (iso) | 2688 | 0.2% | 84.4% | 0.2% |

## target_attribute=currency

overall accuracy: 23.0% (n=9888)
final_score = 1/2(cause + mean(iso)) = 1/2(84.4% + 0.1%) = **42.2%**

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 0.0% | 97.1% | 0.0% |
| capital (iso) | 3600 | 0.2% | 97.3% | 0.2% |
| currency (cause) | 2688 | 84.4% | 84.4% | 0.2% |

## target_attribute=language

overall accuracy: 0.2% (n=9888)

| queried | n | accuracy | matches_source | matches_base |
|---|---|---|---|---|
| calling_code (iso) | 3600 | 0.0% | 97.1% | 0.0% |
| capital (iso) | 3600 | 0.2% | 97.3% | 0.2% |
| currency (iso) | 2688 | 0.2% | 84.4% | 0.2% |
