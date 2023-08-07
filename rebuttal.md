# Rebuttal
can be used in all reply:
The primary focus of this paper is to introduce an innovative approach for leveraging evidential learning in the image-text retrieval task, which enables improved uncertainty estimation. To achieve this, we employ adapters to efficiently fine-tune mainstream pre-trained Vision-and-Language (VL) models, such as CLIP and BLIP. Our method involves a gradual shift from a simple probability distribution (softMax) to a more robust Dirichlet Distribution (evidence) as the model's posterior. Consequently, our model transforms into a deterministic uncertainty estimator.
It is important to note that our method does not primarily aim to improve overall retrieval accuracy. Instead, our key objective is to enhance the model's robustness when confronted with out-of-distribution (OOD) cases. The shift from the probability distribution (softMax) to the Dirichlet Distribution (evidence) effectively addresses the issue of over-confidence in the model's predictions. During inference, such as image-to-text (i2t) or text-to-image (t2i), the final softmax function in traditional models tends to force certain predictions, which works well for in-distribution (ID) cases. However, in OOD cases, this over-confidence becomes problematic. OOD inputs often contain misleading information, but the well-trained model disregards it and provides predictions with unwavering certainty.
To tackle this problem, our approach considers the Dirichlet Distribution (evidence) during inference, allowing the model to weigh all probabilities rather than relying on a single point probability. This consideration of multiple probabilities is crucial in OOD cases, where the matching scores between images and texts may exhibit an even distribution. By incorporating evidence-driven fine-tuning and embracing the uncertainty inherent in the Dirichlet Distribution, our model achieves enhanced robustness in image-text retrieval tasks, particularly when handling challenging out-of-distribution scenarios.

## Reviewer 1
R1.1 The experiment result do not support the claim "significantly improve the performance of the mainstream pre-trained VL models (e.g., CLIP and BLIP) on both in-distribution and out-of-distribution cases for image-text retrieval." In Tab.1, most result of the proposed ELIP is worse than the baseline BLIP+ft.

A1.1 The primary focus of our research is centered around two key objectives: achieving deterministic uncertainty estimation and enhancing model robustness in out-of-distribution (OOD) cases. In Table 1, our method, ELIP, exhibits remarkable performance, surpassing BLIP-ft in both OOD-image and OOD-image&text scenarios. This achievement is particularly noteworthy as we fine-tuned a significantly smaller subset of parameters (65M) compared to BLIP, which utilizes a more complex structure and fine-tunes the entire model. Instead of a direct retrieval accuracy comparison, we adopt a reference paper [1]: <https://arxiv.org/pdf/2212.08044.pdf> suggested by reviewer 5LZy to ensure a more reasonable and objective evaluation. After reading, we found one benchmark named MultiModal Impact score, which can be used to analyze our existing experiment results, since MMI is used to easure the relative performance drop between the ID and OOD performance.
Here is the MMI comparision based on our existing results:

| **Image Retrieval** |      |           |      |      |                 |      |           |           |          |
|---------------------|------|-----------|------|------|-----------------|------|-----------|-----------|----------|
|                     |      | **Clean** |      |      | **Average OOD** |      |           | **MMI**   |          |
|                     | R@1  | R@5       | R@10 | R@1  | R@5             | R@10 | R@1       | R@5       | R@10     |
|     CLIP-zs         | 35.3 | 60.0      | 70.2 | 27.4 | 50.5            | 61.2 | &darr;22.3%     | &darr;15.8%     | &darr;12.9%    |
|     ALBEF-ft        | 60.7 | 84.3      | 90.5 | 47.0 | 71.5            | 80.2 | &darr;22.6%     | &darr;15.2%     | &darr;11.4%    |
|     BLIP-ft         | 64.3 | 85.7      | 91.5 | 51.3 | 74.5            | 82.2 | &darr;20.3%     | &darr;13.0%     | &darr;10.1%    |
|     ELIP            | 60.4 | 83.9      | 90.5 | 49.2 | 74.4            | 83.0 | &darr;**18.6%** | &darr;**11.3%** | &darr;**8.3%** |
| ELIP+               | 63.7 | 85.4      | 91.3 | 51.2 | 74.6            | 82.4 | &darr;**19.6%** | &darr;**12.6%** | &darr;**9.7%** |

| **Text Retrieval** |      |           |      |      |                 |      |           |           |          |
|--------------------|------|-----------|------|------|-----------------|------|-----------|-----------|----------|
|                    |      | **Clean** |      |      | **Average OOD** |      |           | **MMI**   |          |
|                    | R@1  | R@5       | R@10 | R@1  | R@5             | R@10 | R@1       | R@5       | R@10     |
|     CLIP-zs        | 56.0 | 79.6      | 86.9 | 43.0 | 58.4            | 77.8 | &darr;23.3%     | &darr;26.7%     | &darr;10.5%    |
|     ALBEF-ft       | 77.6 | 94.3      | 97.2 | 61.8 | 81.9            | 87.5 | &darr;20.3%     | &darr;13.1%     | &darr;9.9%     |
|     BLIP-ft        | 81.9 | 95.4      | 97.8 | 67.0 | 85.1            | 89.9 | &darr;18.2%     | &darr;10.8%     | &darr;8.1%     |
|     ELIP           | 77.5 | 94.2      | 97.0 | 65.9 | 86.3            | 91.8 | &darr;**15.0%** | &darr;**8.4%**  | &darr;**5.4%** |
| ELIP+              | 81.3 | 95.2      | 97.7 | 66.9 | 85.0            | 89.8 | &darr;**17.7%** | &darr;**10.7%** | &darr;**8.1%** |

We have observed that ELIP and ELIP+ outperform all other baseline models on the MMI benchmark, demonstrating the effectiveness of our method in simple OOD settings.

R1.2 The writing of the approach part is not clear, the main figure (Fig.3) also makes me confused since the locations of all components do not have a clear order (like from left to right, or bottom to up, etc.)

A1.2 ELIP is a double stream encoder-based structured, in Fig.3 you can follow the components clockwisely, starting from the bottom-left corner. 
Specifically, **training**: input (image-text pair) &rarr; image & text encoder &rarr; feature output alignment &rarr; evidencial learning;
**inference**: input (image & text) &rarr; image & text encoder &rarr; feature output alignment &rarr; retrieval & uncertainty estimation.





To validate our method's ability to handle more complex and diverse OOD cases, we followed the same reference and generated different sets of perturbated images and texts.
Experiment 1: We choose three **OOD-image** settings from mentioned in <https://arxiv.org/pdf/2212.08044.pdf>, all results are average on five perturbation levels. The following are Image-Text Retrieval performance comparision between ELIP and baseline models on MS-COCO.

| Image Retrieval |      |           |      |      |      |      |      |      |      |
|-----------------|------|-----------|------|------|------|------|------|------|------|
|                 |      | Blur-Zoom |      |      | Snow |      |      | JPEG |      |
|                 | R@1  | R@5       | R@10 | R@1  | R@5  | R@10 | R@1  | R@5  | R@10 |
|     CLIP-zs     | 26.9 | 50.1      | 61.0 | 26.8 | 50.1 | 61.4 | 35.9 | 60.7 | 70.9 |
|     ALBEF-ft    | 29.2 | 51.3      | 60.9 | 44.9 | 71.0 | 79.9 | 55.3 | 80.0 | 87.4 |
|     BLIP-ft     | 31.8 | 53.4      | 62.5 | 49.7 | 74.5 | 82.8 | **60.1** | **83.0** | 89.5 |
|     ELIP        | **42.6** | **68.4**      | **77.7** | **51.5** | **76.9** | **85.3** | 58.9 | 82.5 | **89.6** |

| Text Retrieval |          |           |          |          |          |          |          |          |          |
|----------------|----------|-----------|----------|----------|----------|----------|----------|----------|----------|
|                |          | Blur-Zoom |          |          | Snow     |          |          | JPEG     |          |
|                | R@1      | R@5       | R@10     | R@1      | R@5      | R@10     | R@1      | R@5      | R@10     |
|     CLIP-zs    | 32.4     | 57.0      | 67.2     | 32.3     | 56.2     | 67.8     | 55.3     | 78.9     | 86.4     |
|     ALBEF-ft   | 29.4     | 51.1      | 60.2     | 51.3     | 76.8     | 84.8     | 71.7     | 91.1     | 95.4     |
|     BLIP-ft    | 30.7     | 52.2      | 61.0     | 58.3     | 80.5     | 87.1     | **77.5** | **93.2** | 96.4     |
|     ELIP       | **38.7** | **65.6**  | **75.6** | **60.3** | **83.7** | **90.6** | 76.4     | 93.1     | **96.4** |

Experiment 2: We choose three **OOD-text** settings from mentioned in <https://arxiv.org/pdf/2212.08044.pdf>, all results are average on five perturbation levels. The following are Image-Text Retrieval performance comparision between ELIP and baseline models on MS-COCO.

