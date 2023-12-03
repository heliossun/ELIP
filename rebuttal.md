# Rebuttal

## Reviewer rBGR
`Q1`: Missing evaluation with GPT-4v.

`A1`: 

## Reviewer G7DB
`Q1`: Since the ELIP+ is initialized by BLIP (line 569), why its performance is lower than BLIP in some settings (table 1). Could you provide a explanation?

`A1`: 

## Reviewer J2CD
`R1`: Has the author tried other more perturbation methods to generate the OOD dataset?

`A1`: We thank the reviewer `j2CD` for the comments. We generate five more OOD cases, and we provide the comparison results below.
>* For each OOD case, we generate five levels of perturbation, and we provide averaged Recall(R@k) over 5 perturbation levels.  

|              |         |          | i2t      |          |          |          | t2i      |          |          |
|--------------|---------|----------|----------|----------|----------|----------|----------|----------|----------|
| Perturbation | Method  | R@1      | R@5      | R@10     | Mean     | R@1      | R@5      | R@10     | Mean     |
|              | CLIP ZS | 47.6     | 71.6     | 80.3     | 66.5     | 34.2     | 58.5     | 69.1     | 53.9     |
|              | CLIP    | 42.4     | 69.9     | 79.9     | 64.1     | 34.9     | 63.3     | 74.9     | 57.7     |
| Shot         | ALBEF   | 66.2     | 86.6     | 92.0     | 81.6     | 52.1     | 77.9     | 85.8     | 71.9     |
|              | BLIP    | 70.1     | 88.2     | 92.8     | 83.7     | 55.2     | 79.2     | 86.5     | 73.7     |
|              | ELIP    | **71.8** | **90.1** | **94.4** | **85.5** | **55.7** | **80.2** | **87.7** | **74.6** |
|              |         |          |          |          |          |          |          |          |          |
|              | CLIP ZS | 40.1     | 65.6     | 75.4     | 60.4     | 30.1     | 54.1     | 64.8     | 49.7     |
|              | CLIP    | 35.6     | 63.0     | 74.3     | 57.6     | 29.8     | 58.3     | 70.7     | 53.0     |
| Impulse      | ALBEF   | 66.0     | 86.8     | 92.1     | 81.6     | 52.1     | 77.9     | 85.8     | 71.9     |
|              | BLIP    | 68.7     | 87.6     | 92.3     | 82.9     | 54.5     | 78.6     | 86.1     | 73.1     |
|              | ELIP    | **72.3** | **90.4** | **94.7** | **85.8** | **56.7** | **81.1** | **88.5** | **75.4** |
|              |         |          |          |          |          |          |          |          |          |
|              | CLIP ZS | 46.5     | 71.3     | 80.0     | 65.9     | 33.7     | 58.3     | 68.8     | 53.6     |
|              | CLIP    | 43.7     | 71.7     | 81.5     | 65.6     | 35.2     | 63.8     | 75.2     | 58.1     |
| Defocus      | ALBEF   | 62.6     | 84.1     | 90.1     | 79.0     | 50.6     | 75.7     | 83.9     | 70.1     |
|              | BLIP    | 68.0     | 87.5     | 92.2     | 82.6     | 54.6     | 78.3     | 85.4     | 72.8     |
|              | ELIP    | **68.3** | **89.1** | **94.2** | **83.9** | **56.0** | **80.4** | **88.0** | **74.8** |
|              |         |          |          |          |          |          |          |          |          |
|              | CLIP ZS | 49.5     | 73.9     | 82.0     | 68.5     | 34.6     | 59.1     | 69.6     | 54.4     |
|              | CLIP    | 36.5     | 65.7     | 77.1     | 59.8     | 36.5     | 65.7     | 77.1     | 59.8     |
| Speckle      | ALBEF   | 69.9     | 89.3     | 94.1     | 84.4     | 54.7     | 80.1     | 87.6     | 74.1     |
|              | BLIP    | **74.4** | **91.5** | 95.0     | **87.0** | **58.4** | **81.6** | **88.5** | **76.2** |
|              | ELIP    | 73.1     | 91.0     | 95.1     | 86.4     | 56.6     | 81.0     | 88.3     | 75.3     |
|              |         |          |          |          |          |          |          |          |          |
|              | CLIP ZS | 36.3     | 60.4     | 70.3     | 55.7     | 27.9     | 51.3     | 61.9     | 47.0     |
|              | CLIP    | 32.4     | 58.3     | 68.9     | 53.2     | 27.3     | 53.8     | 65.7     | 48.9     |
| Pixel        | ALBEF   | 45.9     | 65.7     | 72.7     | 61.4     | 36.3     | 58.9     | 67.5     | 54.2     |
|              | BLIP    | 56.1     | 76.3     | 82.6     | 71.6     | 44.9     | 68.3     | 76.5     | 63.3     |
|              | ELIP    | **67.1** | **88.6** | **93.4** | **83.0** | **54.8** | **79.1** | **86.9** | **73.6** |

From the new experiments, we observed that ELIP presents the best robustness in most OOD cases.



## Reviewer 4


## Reviewer 5


1. We test on five more image perturbation settings. We provide averaged Recall, RSUM, and MMI over five perturbation levels, to evaluate the robustness of each model.
Where RSUM = SUM(i2t(R@1,R@5,R@10)+t2i(R@1,R@5,R@10))



| Method   | Clean | Shot  | Impulse | Speckle | Defocus | Pixel | **ave** | MMI |
|----------|-------|-------|---------|---------|---------|-------|---------|-----|
| CLIP ZS  | 394.5 | 361.2 | 330.2   | 368.7   | 358.7   | 308.2 |  345.4  |12.4%|
| CLIP FT  | 420.5 | 365.3 | 331.7   | 381.5   | 371.0   | 306.4 | 351.2   |16.5%|
| ALBEF FT | 504.6 | 460.6 | 460.3   | 376.4   | 447.1   | 347.0 | 418.3   |17.1%|
| BLIP FT  | 516.6 | 472.1 | 467.7   | 489.5   | 466.1   | 404.7 | 460.2   |10.9%|
| ELIP     | 503.5 | 480.0 | 483.7   | 485.0   |  476.2  | 469.8 | 478.9   |4.9%|

| **Uncertainty**    |      |      | i2t  |      |      |      |      |      |      | t2i  |      |      |
|--------------------|------|------|------|------|------|------|------|------|------|------|------|------|
| Perturbation/Level | 1    | 2    | 3    | 4    | 5    | avg  | 1    | 2    | 3    | 4    | 5    | avg  |
| Shot               | 0.43 | 0.44 | 0.46 | 0.48 | 0.51 |      | 0.65 | 0.66 | 0.67 | 0.69 | 0.72 |      |
| Impulse            | 0.44 | 0.45 | 0.46 | 0.48 | 0.51 |      | 0.66 | 0.66 | 0.67 | 0.69 | 0.72 |      |
| Defocus            | 0.45 | 0.46 | 0.49 | 0.52 | 0.56 |      | 0.67 | 0.68 | 0.72 | 0.76 | 0.79 |      |
| Speckle            | 0.43 | 0.44 | 0.45 | 0.46 | 0.48 |      | 0.66 | 0.66 | 0.67 | 0.68 | 0.70 |      |
| Pixel              | 0.45 | 0.46 | 0.49 | 0.53 | 0.57 |      | 0.67 | 0.68 | 0.70 | 0.75 | 0.80 |      |
| Clean              | -    | -    | -    | -    | -    | 0.43 | -    | -    | -    | -    | -    | 0.65 |


