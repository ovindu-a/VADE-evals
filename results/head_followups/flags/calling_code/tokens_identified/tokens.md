# Token identification

Rankings are frozen from the image-patched last prompt token. Image coordinates are zero-based.
Attention is softmax probability; contribution is the norm of A[q,k] V[k] W_O for this head.

## Row 3458: ZM → AO (calling_code_prefill_v3)

### Head 21.1

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 79 | image[5,4] | object | 0.34961 | 7.75687 |
| 153 | image[11,6] | image_background | 0.21387 | 5.81854 |
| 92 | image[6,5] | object | 0.12988 | 3.11665 |
| 21 | image[0,6] | image_background | 0.08887 | 2.10710 |
| 2 | '\n' | text | 0.05176 | 0.09331 |
| 33 | image[1,6] | image_background | 0.04883 | 1.33413 |
| 120 | image[8,9] | image_background | 0.02612 | 0.45095 |
| 125 | image[9,2] | image_background | 0.02148 | 0.45595 |

### Head 21.5

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 92 | image[6,5] | object | 0.35352 | 7.71385 |
| 79 | image[5,4] | object | 0.22461 | 4.52095 |
| 153 | image[11,6] | image_background | 0.11719 | 2.86682 |
| 2 | '\n' | text | 0.09619 | 0.20373 |
| 21 | image[0,6] | image_background | 0.04321 | 0.91925 |
| 125 | image[9,2] | image_background | 0.01990 | 0.38958 |
| 120 | image[8,9] | image_background | 0.01953 | 0.30620 |
| 33 | image[1,6] | image_background | 0.01855 | 0.45590 |

### Head 22.19

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 92 | image[6,5] | object | 0.20020 | 4.32017 |
| 79 | image[5,4] | object | 0.16016 | 3.47243 |
| 2 | '\n' | text | 0.11475 | 0.17569 |
| 153 | image[11,6] | image_background | 0.10791 | 2.49110 |
| 21 | image[0,6] | image_background | 0.07422 | 1.81345 |
| 33 | image[1,6] | image_background | 0.03784 | 0.93999 |
| 175 | '<\|im_start\|>' | special | 0.03113 | 0.60463 |
| 183 | ' +' | text | 0.01489 | 0.24377 |

### Head 23.17

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 79 | image[5,4] | object | 0.31250 | 12.07721 |
| 21 | image[0,6] | image_background | 0.16211 | 5.81953 |
| 153 | image[11,6] | image_background | 0.15332 | 5.89102 |
| 92 | image[6,5] | object | 0.13086 | 5.09848 |
| 2 | '\n' | text | 0.12109 | 0.10130 |
| 33 | image[1,6] | image_background | 0.03809 | 1.51338 |
| 66 | image[4,3] | image_background | 0.00854 | 0.35717 |
| 183 | ' +' | text | 0.00842 | 0.24624 |

### Head 23.4

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 79 | image[5,4] | object | 0.20215 | 4.92302 |
| 21 | image[0,6] | image_background | 0.17676 | 4.80121 |
| 92 | image[6,5] | object | 0.10938 | 2.84757 |
| 153 | image[11,6] | image_background | 0.08350 | 2.20953 |
| 2 | '\n' | text | 0.06982 | 0.06946 |
| 183 | ' +' | text | 0.04492 | 0.83988 |
| 33 | image[1,6] | image_background | 0.04126 | 1.13861 |
| 96 | image[6,9] | image_background | 0.02637 | 0.43263 |

### Head 23.3

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 79 | image[5,4] | object | 0.31250 | 7.17674 |
| 92 | image[6,5] | object | 0.21582 | 5.54656 |
| 21 | image[0,6] | image_background | 0.17871 | 4.54235 |
| 153 | image[11,6] | image_background | 0.15820 | 3.88502 |
| 33 | image[1,6] | image_background | 0.04492 | 1.17654 |
| 2 | '\n' | text | 0.02991 | 0.03558 |
| 125 | image[9,2] | image_background | 0.00891 | 0.28268 |
| 175 | '<\|im_start\|>' | special | 0.00867 | 0.18512 |

### Head 23.7

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 2 | '\n' | text | 0.34375 | 0.26180 |
| 79 | image[5,4] | object | 0.17773 | 7.40767 |
| 153 | image[11,6] | image_background | 0.17773 | 7.50107 |
| 21 | image[0,6] | image_background | 0.06885 | 2.63601 |
| 92 | image[6,5] | object | 0.05078 | 1.94581 |
| 33 | image[1,6] | image_background | 0.03320 | 1.41666 |
| 66 | image[4,3] | image_background | 0.02014 | 0.89065 |
| 120 | image[8,9] | image_background | 0.01685 | 0.67328 |

### Head 23.6

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 2 | '\n' | text | 0.72656 | 0.94897 |
| 79 | image[5,4] | object | 0.03467 | 0.81030 |
| 153 | image[11,6] | image_background | 0.02576 | 0.63658 |
| 175 | '<\|im_start\|>' | special | 0.01404 | 0.30581 |
| 183 | ' +' | text | 0.01178 | 0.22135 |
| 163 | ' country' | text | 0.01111 | 0.15043 |
| 167 | ' the' | text | 0.01074 | 0.15359 |
| 21 | image[0,6] | image_background | 0.01044 | 0.27440 |

## Row 1577: HN → VN (calling_code_prefill_v6)

### Head 21.1

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 2 | '\n' | text | 0.36328 | 0.65491 |
| 21 | image[0,6] | image_background | 0.12793 | 2.61012 |
| 91 | image[6,4] | object | 0.06836 | 1.49314 |
| 153 | image[11,6] | image_background | 0.06689 | 1.51061 |
| 65 | image[4,2] | image_background | 0.04419 | 0.93194 |
| 97 | image[6,10] | image_background | 0.03760 | 0.71825 |
| 169 | ' country' | text | 0.03296 | 0.51327 |
| 92 | image[6,5] | object | 0.02698 | 0.52305 |

### Head 21.5

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 2 | '\n' | text | 0.35156 | 0.74459 |
| 91 | image[6,4] | object | 0.20312 | 4.12930 |
| 71 | image[4,8] | image_background | 0.06689 | 1.22124 |
| 21 | image[0,6] | image_background | 0.03809 | 0.72248 |
| 94 | image[6,7] | object | 0.03662 | 0.59603 |
| 92 | image[6,5] | object | 0.02466 | 0.44075 |
| 153 | image[11,6] | image_background | 0.02417 | 0.50503 |
| 65 | image[4,2] | image_background | 0.02380 | 0.46528 |

### Head 22.19

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 2 | '\n' | text | 0.09912 | 0.15177 |
| 91 | image[6,4] | object | 0.09033 | 1.70248 |
| 21 | image[0,6] | image_background | 0.08447 | 1.72595 |
| 175 | '\n' | text | 0.05347 | 0.49975 |
| 169 | ' country' | text | 0.04883 | 0.64042 |
| 180 | ' +' | text | 0.04419 | 0.51253 |
| 161 | ' country' | text | 0.04150 | 0.45432 |
| 168 | ' this' | text | 0.03833 | 0.45552 |

### Head 23.17

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 2 | '\n' | text | 0.29688 | 0.24834 |
| 153 | image[11,6] | image_background | 0.16406 | 5.67039 |
| 21 | image[0,6] | image_background | 0.12109 | 4.20691 |
| 65 | image[4,2] | image_background | 0.09229 | 3.43161 |
| 82 | image[5,7] | object | 0.06348 | 1.78275 |
| 91 | image[6,4] | object | 0.04907 | 1.62253 |
| 149 | image[11,2] | image_background | 0.03516 | 1.15161 |
| 180 | ' +' | text | 0.03516 | 1.07563 |

### Head 23.4

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 21 | image[0,6] | image_background | 0.19043 | 5.04541 |
| 91 | image[6,4] | object | 0.14258 | 3.76950 |
| 94 | image[6,7] | object | 0.07471 | 2.01385 |
| 97 | image[6,10] | image_background | 0.07129 | 1.87049 |
| 180 | ' +' | text | 0.05225 | 0.75163 |
| 153 | image[11,6] | image_background | 0.04663 | 1.14329 |
| 23 | image[0,8] | image_background | 0.04321 | 1.05287 |
| 2 | '\n' | text | 0.03760 | 0.03740 |

### Head 23.3

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 21 | image[0,6] | image_background | 0.32617 | 7.84465 |
| 2 | '\n' | text | 0.12500 | 0.14873 |
| 91 | image[6,4] | object | 0.12012 | 2.96034 |
| 153 | image[11,6] | image_background | 0.10693 | 2.35638 |
| 65 | image[4,2] | image_background | 0.08691 | 2.00230 |
| 71 | image[4,8] | image_background | 0.04346 | 0.96494 |
| 149 | image[11,2] | image_background | 0.03271 | 0.66692 |
| 94 | image[6,7] | object | 0.02417 | 0.61898 |

### Head 23.7

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 2 | '\n' | text | 0.53516 | 0.40758 |
| 153 | image[11,6] | image_background | 0.12598 | 4.40836 |
| 65 | image[4,2] | image_background | 0.09277 | 3.39159 |
| 21 | image[0,6] | image_background | 0.05444 | 1.83555 |
| 149 | image[11,2] | image_background | 0.04688 | 1.59754 |
| 91 | image[6,4] | object | 0.01733 | 0.55253 |
| 49 | image[2,10] | image_background | 0.00903 | 0.20448 |
| 94 | image[6,7] | object | 0.00806 | 0.24188 |

### Head 23.6

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 2 | '\n' | text | 0.26562 | 0.34694 |
| 91 | image[6,4] | object | 0.06982 | 1.79200 |
| 21 | image[0,6] | image_background | 0.05444 | 1.34128 |
| 94 | image[6,7] | object | 0.04150 | 1.11856 |
| 169 | ' country' | text | 0.03931 | 0.62829 |
| 153 | image[11,6] | image_background | 0.03540 | 0.79402 |
| 82 | image[5,7] | object | 0.03467 | 0.79689 |
| 168 | ' this' | text | 0.03320 | 0.53840 |

## Row 3104: UA → MA (calling_code_prefill_v3)

### Head 21.1

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 2 | '\n' | text | 0.34375 | 0.61970 |
| 69 | image[4,6] | image_background | 0.18359 | 4.22480 |
| 81 | image[5,6] | object | 0.05347 | 1.20489 |
| 167 | ' the' | text | 0.04199 | 0.39402 |
| 93 | image[6,6] | object | 0.03955 | 1.04623 |
| 57 | image[3,6] | image_background | 0.03516 | 0.83922 |
| 49 | image[2,10] | image_background | 0.02734 | 0.54892 |
| 82 | image[5,7] | object | 0.02454 | 0.57598 |

### Head 21.5

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 81 | image[5,6] | object | 0.25000 | 5.33154 |
| 2 | '\n' | text | 0.20312 | 0.43021 |
| 69 | image[4,6] | image_background | 0.15527 | 3.22469 |
| 93 | image[6,6] | object | 0.10498 | 2.57045 |
| 57 | image[3,6] | image_background | 0.04883 | 1.05614 |
| 49 | image[2,10] | image_background | 0.04102 | 0.74707 |
| 119 | image[8,8] | image_background | 0.02271 | 0.40542 |
| 82 | image[5,7] | object | 0.02197 | 0.48794 |

### Head 22.19

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 2 | '\n' | text | 0.09961 | 0.15252 |
| 69 | image[4,6] | image_background | 0.07520 | 1.65020 |
| 81 | image[5,6] | object | 0.07373 | 1.78250 |
| 175 | '<\|im_start\|>' | special | 0.07373 | 1.17318 |
| 163 | ' country' | text | 0.07178 | 0.83180 |
| 183 | ' +' | text | 0.05640 | 0.72509 |
| 177 | '\n' | text | 0.04346 | 0.40759 |
| 149 | image[11,2] | image_background | 0.03735 | 0.48068 |

### Head 23.17

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 69 | image[4,6] | image_background | 0.36133 | 11.28802 |
| 57 | image[3,6] | image_background | 0.13867 | 4.33135 |
| 2 | '\n' | text | 0.13477 | 0.11273 |
| 81 | image[5,6] | object | 0.08789 | 2.60687 |
| 82 | image[5,7] | object | 0.04980 | 1.47186 |
| 49 | image[2,10] | image_background | 0.04785 | 1.39955 |
| 153 | image[11,6] | image_background | 0.03516 | 1.23976 |
| 93 | image[6,6] | object | 0.02942 | 1.01221 |

### Head 23.4

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 81 | image[5,6] | object | 0.17676 | 4.45168 |
| 69 | image[4,6] | image_background | 0.17285 | 3.92299 |
| 93 | image[6,6] | object | 0.12695 | 4.08819 |
| 101 | image[7,2] | image_background | 0.10010 | 2.31492 |
| 49 | image[2,10] | image_background | 0.05859 | 1.20449 |
| 183 | ' +' | text | 0.05200 | 0.79653 |
| 175 | '<\|im_start\|>' | special | 0.03906 | 0.68287 |
| 2 | '\n' | text | 0.03833 | 0.03813 |

### Head 23.3

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 69 | image[4,6] | image_background | 0.38477 | 8.45034 |
| 81 | image[5,6] | object | 0.16797 | 4.26489 |
| 93 | image[6,6] | object | 0.10986 | 3.49549 |
| 57 | image[3,6] | image_background | 0.07568 | 1.61466 |
| 49 | image[2,10] | image_background | 0.06006 | 1.14433 |
| 2 | '\n' | text | 0.05591 | 0.06652 |
| 153 | image[11,6] | image_background | 0.04785 | 0.97371 |
| 101 | image[7,2] | image_background | 0.02075 | 0.41973 |

### Head 23.7

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 2 | '\n' | text | 0.55078 | 0.41948 |
| 69 | image[4,6] | image_background | 0.12012 | 4.31760 |
| 153 | image[11,6] | image_background | 0.04980 | 2.02455 |
| 57 | image[3,6] | image_background | 0.04346 | 1.57581 |
| 105 | image[7,6] | image_background | 0.02051 | 0.64400 |
| 81 | image[5,6] | object | 0.01721 | 0.56446 |
| 93 | image[6,6] | object | 0.01471 | 0.55687 |
| 49 | image[2,10] | image_background | 0.01379 | 0.43251 |

### Head 23.6

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 2 | '\n' | text | 0.19434 | 0.25383 |
| 69 | image[4,6] | image_background | 0.08643 | 1.98423 |
| 167 | ' the' | text | 0.07861 | 0.84307 |
| 175 | '<\|im_start\|>' | special | 0.06543 | 1.17315 |
| 163 | ' country' | text | 0.04858 | 0.69733 |
| 93 | image[6,6] | object | 0.04297 | 1.40371 |
| 183 | ' +' | text | 0.03857 | 0.59359 |
| 162 | ' the' | text | 0.03540 | 0.29353 |

## Row 1722: IN → ZM (calling_code_prefill_v1)

### Head 21.1

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 105 | image[7,6] | image_background | 0.55469 | 15.49967 |
| 2 | '\n' | text | 0.20703 | 0.37323 |
| 95 | image[6,8] | image_background | 0.03687 | 0.93311 |
| 34 | image[1,7] | image_background | 0.03467 | 0.83255 |
| 73 | image[4,10] | image_background | 0.01526 | 0.29935 |
| 81 | image[5,6] | object | 0.01392 | 0.33143 |
| 133 | image[9,10] | image_background | 0.01001 | 0.16654 |
| 167 | ' the' | text | 0.00885 | 0.08633 |

### Head 21.5

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 105 | image[7,6] | image_background | 0.37891 | 9.33759 |
| 2 | '\n' | text | 0.24707 | 0.52328 |
| 80 | image[5,5] | object | 0.09717 | 1.69400 |
| 95 | image[6,8] | image_background | 0.07324 | 1.64742 |
| 34 | image[1,7] | image_background | 0.02588 | 0.54835 |
| 167 | ' the' | text | 0.01758 | 0.16713 |
| 168 | ' country' | text | 0.01483 | 0.19898 |
| 125 | image[9,2] | image_background | 0.01270 | 0.15818 |

### Head 22.19

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 105 | image[7,6] | image_background | 0.31641 | 7.45046 |
| 2 | '\n' | text | 0.09521 | 0.14579 |
| 34 | image[1,7] | image_background | 0.05249 | 1.29820 |
| 21 | image[0,6] | image_background | 0.05054 | 1.37182 |
| 182 | ' +' | text | 0.04541 | 0.66026 |
| 167 | ' the' | text | 0.04077 | 0.37923 |
| 80 | image[5,5] | object | 0.04028 | 0.67318 |
| 168 | ' country' | text | 0.02441 | 0.29512 |

### Head 23.17

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 105 | image[7,6] | image_background | 0.76562 | 28.92210 |
| 2 | '\n' | text | 0.11035 | 0.09231 |
| 34 | image[1,7] | image_background | 0.02051 | 0.66396 |
| 92 | image[6,5] | object | 0.01019 | 0.32006 |
| 94 | image[6,7] | object | 0.00958 | 0.33862 |
| 182 | ' +' | text | 0.00934 | 0.27422 |
| 79 | image[5,4] | object | 0.00830 | 0.23576 |
| 95 | image[6,8] | image_background | 0.00830 | 0.18980 |

### Head 23.4

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 105 | image[7,6] | image_background | 0.50391 | 15.00877 |
| 182 | ' +' | text | 0.08643 | 1.50334 |
| 81 | image[5,6] | object | 0.05273 | 1.15372 |
| 141 | image[10,6] | image_background | 0.04321 | 1.02528 |
| 125 | image[9,2] | image_background | 0.04004 | 0.74642 |
| 34 | image[1,7] | image_background | 0.03345 | 0.91864 |
| 133 | image[9,10] | image_background | 0.03076 | 0.74913 |
| 2 | '\n' | text | 0.03027 | 0.03012 |

### Head 23.3

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 105 | image[7,6] | image_background | 0.87891 | 26.04500 |
| 81 | image[5,6] | object | 0.03003 | 0.65145 |
| 2 | '\n' | text | 0.02478 | 0.02948 |
| 34 | image[1,7] | image_background | 0.02051 | 0.54591 |
| 95 | image[6,8] | image_background | 0.00772 | 0.20718 |
| 133 | image[9,10] | image_background | 0.00610 | 0.14607 |
| 80 | image[5,5] | object | 0.00267 | 0.05334 |
| 182 | ' +' | text | 0.00243 | 0.04078 |

### Head 23.7

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 2 | '\n' | text | 0.60547 | 0.46113 |
| 105 | image[7,6] | image_background | 0.13574 | 5.44259 |
| 163 | ' international' | text | 0.05591 | 1.89100 |
| 34 | image[1,7] | image_background | 0.01855 | 0.59910 |
| 177 | '\n' | text | 0.01648 | 0.15375 |
| 178 | 'The' | text | 0.01349 | 0.19249 |
| 79 | image[5,4] | object | 0.01227 | 0.37237 |
| 176 | 'assistant' | text | 0.00873 | 0.14488 |

### Head 23.6

| Position | Token / image patch | Group | Attention | Contribution |
|---|---|---|---:|---:|
| 2 | '\n' | text | 0.41992 | 0.54847 |
| 167 | ' the' | text | 0.11621 | 1.23977 |
| 105 | image[7,6] | image_background | 0.07324 | 2.01301 |
| 182 | ' +' | text | 0.05103 | 0.89235 |
| 168 | ' country' | text | 0.02002 | 0.28812 |
| 163 | ' international' | text | 0.01941 | 0.31760 |
| 141 | image[10,6] | image_background | 0.01794 | 0.45286 |
| 165 | ' code' | text | 0.01770 | 0.25571 |

