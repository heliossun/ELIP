# Rebuttal
Reply to all:
The primary focus of this paper is to introduce an innovative approach for leveraging evidential learning in the image-text retrieval task, which enables improved uncertainty estimation. To achieve this, we employ adapters to efficiently fine-tune mainstream pre-trained Vision-and-Language (VL) models, such as CLIP and BLIP. Our method involves a gradual shift from a simple probability distribution (softMax) to a more robust Dirichlet Distribution (evidence) as the model's posterior. Consequently, our model transforms into a deterministic uncertainty estimator.
It is important to note that our method does not primarily aim to improve overall retrieval accuracy. Instead, our key objective is to enhance the model's robustness when confronted with out-of-distribution (OOD) cases. The shift from the probability distribution (softMax) to the Dirichlet Distribution (evidence) effectively addresses the issue of over-confidence in the model's predictions. During inference, such as image-to-text (i2t) or text-to-image (t2i), the final softmax function in traditional models tends to force certain predictions, which works well for in-distribution (ID) cases. However, in OOD cases, this over-confidence becomes problematic. OOD inputs often contain misleading information, but the well-trained model disregards it and provides predictions with unwavering certainty.
To tackle this problem, our approach considers the Dirichlet Distribution (evidence) during inference, allowing the model to weigh all probabilities rather than relying on a single point probability. This consideration of multiple probabilities is crucial in OOD cases, where the matching scores between images and texts may exhibit an even distribution. By incorporating evidence-driven fine-tuning and embracing the uncertainty inherent in the Dirichlet Distribution, our model achieves enhanced robustness in image-text retrieval tasks, particularly when handling challenging out-of-distribution scenarios.

## Reviewer 1
### Question:
1. The experiment result do not support the claim "significantly improve the performance of the mainstream pre-trained VL models (e.g., CLIP and BLIP) on both in-distribution and out-of-distribution cases for image-text retrieval." In Tab.1, most result of the proposed ELIP is worse than the baseline BLIP+ft.

### Reply:
1. The primary focus of our research is centered around two key objectives: achieving deterministic uncertainty estimation and enhancing model robustness in out-of-distribution (OOD) cases. In Table 1, our method, ELIP, exhibits remarkable performance, surpassing BLIP-ft in both OOD-image and OOD-image&text scenarios. This achievement is particularly noteworthy as we fine-tuned a significantly smaller subset of parameters (65M) compared to BLIP, which utilizes a more complex structure and fine-tunes the entire model. Instead of a direct retrieval accuracy comparison, we adopt a reference paper [1]: <https://arxiv.org/pdf/2212.08044.pdf> suggested by reviewer 5LZy to ensure a more reasonable and objective evaluation. After reading, we found one benchmark named MultiModal Impact score, which can be used to analyze our existing experiment results, since MMI is used to easure the relative performance drop between the ID and OOD performance.
Here is the new table:

| Image retrieval |             |              |             |   |   |   |
|-----------------|-------------|--------------|-------------|---|---|---|
|                 |             |     Clean    |             |   |   |   |
|                 |      R1     |       R5     |      R10    |   |   |   |
|     CLIP        |     35.3    |       60     |     70.2    |   |   |   |
|     BLIP        |     56.9    |      80.8    |     87.9    |   |   |   |
|     ALBEF       |     60.7    |      84.3    |     90.5    |   |   |   |
|     BLIP-ft     |     64.3    |      85.7    |     91.5    |   |   |   |
|     ELIP        |     60.4    |      83.9    |     90.5    |   |   |   |
|     ELIP+       |     63.7    |      85.4    |     91.3    |   |   |   |

