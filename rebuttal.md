# Rebuttal
can be used in all reply:
The primary focus of this paper is to introduce an innovative approach for leveraging evidential learning in the image-text retrieval task, which enables improved uncertainty estimation. To achieve this, we employ adapters to efficiently fine-tune mainstream pre-trained Vision-and-Language (VL) models, such as CLIP and BLIP. Our method involves a gradual shift from a simple probability distribution (softMax) to a more robust Dirichlet Distribution (evidence) as the model's posterior. Consequently, our model transforms into a deterministic uncertainty estimator.
It is important to note that our method does not primarily aim to improve overall retrieval accuracy. Instead, our key objective is to enhance the model's robustness when confronted with out-of-distribution (OOD) cases. The shift from the probability distribution (softMax) to the Dirichlet Distribution (evidence) effectively addresses the issue of over-confidence in the model's predictions. During inference, such as image-to-text (i2t) or text-to-image (t2i), the final softmax function in traditional models tends to force certain predictions, which works well for in-distribution (ID) cases. However, in OOD cases, this over-confidence becomes problematic. OOD inputs often contain misleading information, but the well-trained model disregards it and provides predictions with unwavering certainty.
To tackle this problem, our approach considers the Dirichlet Distribution (evidence) during inference, allowing the model to weigh all probabilities rather than relying on a single point probability. This consideration of multiple probabilities is crucial in OOD cases, where the matching scores between images and texts may exhibit an even distribution. By incorporating evidence-driven fine-tuning and embracing the uncertainty inherent in the Dirichlet Distribution, our model achieves enhanced robustness in image-text retrieval tasks, particularly when handling challenging out-of-distribution scenarios.

## Reviewer 1


R1.1 The experiment result do not support the claim "significantly improve the performance of the mainstream pre-trained VL models (e.g., CLIP and BLIP) on both in-distribution and out-of-distribution cases for image-text retrieval." In Tab.1, most result of the proposed ELIP is worse than the baseline BLIP+ft.

A1.1 **Robustness evaluation:** We agree with the Reviewer's comment, but we want to reemphasize that the primary focus of our research is centered around two key objectives: achieving deterministic uncertainty estimation and enhancing model robustness in out-of-distribution (OOD) cases. In Table 1 (main draft), our method, ELIP, exhibits remarkable performance, surpassing BLIP-ft in both OOD-image and OOD-image&text scenarios. This achievement is particularly noteworthy as we fine-tuned a significantly smaller subset of parameters (65M) compared to BLIP, which utilizes a more complex structure and fine-tunes the entire model. 

In addition, we adopt a reference paper [1] suggested by reviewer 5LZy to further our experiment. After reading, we found the benchmark MultiModal Impact score (MMI) can be used to analyze our existing experiment results, since **MMI** is used to easure the relative performance drop between the ID and OOD performance, this benchmark will ensure a more reasonable and objective evaluation.

\* **In the following table, we grab the clean and average OOD retrieval resuls from Table 1(main draft) directly**. In addition, we provide the **MMI** comparision based on these results:

| **Image Retrieval** |      |           |      |      |                 |      |           |           |          |
|---------------------|------|-----------|------|------|-----------------|------|-----------|-----------|----------|
|                     |      | Clean |      |      | Average OOD |      |           | **MMI**   |          |
|                     | R@1  | R@5       | R@10 | R@1  | R@5             | R@10 | R@1       | R@5       | R@10     |
|     CLIP-zs         | 35.3 | 60.0      | 70.2 | 27.4 | 50.5            | 61.2 | &darr;22.3%     | &darr;15.8%     | &darr;12.9%    |
|     ALBEF-ft        | 60.7 | 84.3      | 90.5 | 47.0 | 71.5            | 80.2 | &darr;22.6%     | &darr;15.2%     | &darr;11.4%    |
|     BLIP-ft         | 64.3 | 85.7      | 91.5 | 51.3 | 74.5            | 82.2 | &darr;20.3%     | &darr;13.0%     | &darr;10.1%    |
|     ELIP            | 60.4 | 83.9      | 90.5 | 49.2 | 74.4            | 83.0 | &darr;**18.6%** | &darr;**11.3%** | &darr;**8.3%** |
| ELIP+               | 63.7 | 85.4      | 91.3 | 51.2 | 74.6            | 82.4 | &darr;**19.6%** | &darr;**12.6%** | &darr;**9.7%** |

| **Text Retrieval** |      |           |      |      |                 |      |           |           |          |
|--------------------|------|-----------|------|------|-----------------|------|-----------|-----------|----------|
|                    |      | Clean |      |      | Average OOD |      |           | **MMI**   |          |
|                    | R@1  | R@5       | R@10 | R@1  | R@5             | R@10 | R@1       | R@5       | R@10     |
|     CLIP-zs        | 56.0 | 79.6      | 86.9 | 43.0 | 58.4            | 77.8 | &darr;23.3%     | &darr;26.7%     | &darr;10.5%    |
|     ALBEF-ft       | 77.6 | 94.3      | 97.2 | 61.8 | 81.9            | 87.5 | &darr;20.3%     | &darr;13.1%     | &darr;9.9%     |
|     BLIP-ft        | 81.9 | 95.4      | 97.8 | 67.0 | 85.1            | 89.9 | &darr;18.2%     | &darr;10.8%     | &darr;8.1%     |
|     ELIP           | 77.5 | 94.2      | 97.0 | 65.9 | 86.3            | 91.8 | &darr;**15.0%** | &darr;**8.4%**  | &darr;**5.4%** |
| ELIP+              | 81.3 | 95.2      | 97.7 | 66.9 | 85.0            | 89.8 | &darr;**17.7%** | &darr;**10.7%** | &darr;**8.1%** |

We have observed that ELIP and ELIP+ outperform all other baseline models on the MMI benchmark, demonstrating the effectiveness of our method in simple OOD settings.

R1.2 The writing of the approach part is not clear, the main figure (Fig.3) also makes me confused since the locations of all components do not have a clear order (like from left to right, or bottom to up, etc.)

A1.2 **The writing of approach wasn't clear:** We appreciate the Reviewers for providing this good suggestion, which gives us a chance to re-summarize our approach and we will clarify this in our final version. The primary focus of this paper is to introduce an innovative approach for leveraging evidential learning in the image-text retrieval task, which enables improved uncertainty estimation. To achieve this, we employ adapters to efficiently fine-tune mainstream pre-trained Vision-and-Language (VL) models, such as CLIP and BLIP. Our method involves a gradual shift from a simple probability distribution (softMax) to a more robust Dirichlet Distribution (evidence) as the model's posterior. Consequently, our model transforms into a deterministic uncertainty estimator.

It is important to note that our method does not primarily aim to improve overall retrieval accuracy. Instead, our key objective is to enhance the model's robustness when confronted with out-of-distribution (OOD) cases. The shift from the probability distribution (softMax) to the Dirichlet Distribution (evidence) effectively addresses the issue of over-confidence in the model's predictions. During inference, such as image-to-text (i2t) or text-to-image (t2i), the final softmax function in traditional models tends to force certain predictions, which works well for in-distribution (ID) cases. However, in OOD cases, this over-confidence becomes problematic. OOD inputs often contain misleading information, but the well-trained model disregards it and provides predictions with unwavering certainty.

To tackle this problem, our approach considers the Dirichlet Distribution (evidence) during inference, allowing the model to weigh all probabilities rather than relying on a single point probability. This consideration of multiple probabilities is crucial in OOD cases, where the matching scores between images and texts may exhibit an even distribution. By incorporating evidence-driven fine-tuning and embracing the uncertainty inherent in the Dirichlet Distribution, our model achieves enhanced robustness in image-text retrieval tasks, particularly when handling challenging out-of-distribution scenarios.

**Durection for Figure 3:** Please follow the components clockwisely, starting from the bottom-left corner, we provide the working-flow of ELIP (training & testing) below, we hope this introduction can assist you have a better reading of our Figure 3. 

**Training**: input (image-text pair) &rarr; image & text encoder &rarr; feature output alignment &rarr; evidencial learning;

**Test**: input (image & text) &rarr; image & text encoder &rarr; feature output alignment (retrieval) &rarr; retrieval & uncertainty estimation.

[1]: <https://arxiv.org/pdf/2212.08044.pdf> 
Qiu, Jielin et al. “Are Multimodal Models Robust to Image and Text Perturbations?” ArXiv abs/2212.08044 (2022): n. pag.

## Reviewer 2

We appreciate the Reviewer's approval and constructive suggestions for us to improve our work. We make the response as below.

R2.1 The authors may want to re-evaluate the contributions of this paper. The technical contribution seems limited, both adapters and evidential loss are well-explored and readily available in the literature.

A2.1 **Lack of technical contribution:** We admit the Review's comment. In the realm of single-domain research, both adapters and evidential loss have been extensively studied, each in their respective contexts. However, in the ELIP framework, we have testes the remarkable robustness of adapters in the context of multi-modal learning, and we have witnessed their exceptional adaptability when faced with new knowledge, especially in the form of evidential information.
Intriguingly, the majority of prior work has primarily employed evidential loss in tasks like classification and regression. However, in our case, we stand out as pioneers in introducing evidential learning to the realm of cross-modal learning. It's important to note that our approach isn't a mere application of these established techniques. Instead, we have undertaken extensive research to understand how to effectively integrate evidential loss within the alignment framework, yielding innovative results.

We want to reemphasize that in essence, ELIP takes the strengths of adapters and evidential loss and leverages them in a unique and synergistic way in the multi-modal and cross-modal learning landscape. Our endeavor encompasses both adaptivity to diverse knowledge forms and a novel application of evidential learning in alignment tasks. This amalgamation stems from rigorous research that goes beyond the surface and delves deep into the intricacies of effectively employing evidential loss within the alignment context. 

R2.2 The authors may want to explain why the proposed approach performs worse than the baseline BLIP+ft in many settings (see Table 1).

A2.2 **ELIP v. VLIP-ft:** We believe this question can be explain from another perspective. We want to reemphasize that our research centers around achieving two primary objectives: deterministic uncertainty estimation and bolstering model robustness in out-of-distribution (OOD) scenarios. In Table 1 9main draft), our method, ELIP, exhibits exceptional performance that outshines BLIP-ft in both OOD-image and OOD-image&text scenarios. Notably, this achievement is magnified by the fact that we fine-tuned a substantially smaller subset of parameters (65M) in comparison to BLIP, which employs a more intricate architecture and conducts fine-tuning across the entire model.

In addition, we adopt a reference paper [1] suggested by reviewer 5LZy to further our experiment. After reading, we found the benchmark MultiModal Impact score can be used to analyze our existing experiment results, since **MMI** is used to easure the relative performance drop between the ID and OOD performance, this benchmark will ensure a more reasonable and objective evaluation.

Incorporating the MMI score, we gain a more comprehensive insight into the effectiveness of our approach. ELIP's ability to excel across various scenarios, combined with our resource-efficient parameter fine-tuning strategy, highlights the significance of our contributions in deterministic uncertainty estimation and model robustness enhancement.

\* **In the following table, we grab the clean and average OOD retrieval resuls from Table 1(main draft) directly.** In addition, we provide the MMI comparision based on these results:

| **Image Retrieval** |      |           |      |      |                 |      |           |           |          |
|---------------------|------|-----------|------|------|-----------------|------|-----------|-----------|----------|
|                     |      | Clean |      |      | Average OOD |      |           | **MMI**   |          |
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

R2.3 The author may want to further explain why the evidential loss seems to have little effect in the ablation study (see Table 3), which is the key innovation of this paper.

A2.3 **Effectiveness evaluation:** We agree with the Review's concer, but we want to reemphasize it is important to note that our method does not primarily aim to improve overall retrieval accuracy. Instead, our key objective is to enhance the model's robustness when confronted with out-of-distribution (OOD) cases. The shift from the probability distribution (softMax) to the Dirichlet Distribution (evidence) effectively addresses the issue of over-confidence in the model's predictions. During inference, such as image-to-text (i2t) or text-to-image (t2i), the final softmax function in traditional models tends to force certain predictions, which works well for in-distribution (ID) cases. However, in OOD cases, this over-confidence becomes problematic. OOD inputs often contain misleading information, but the well-trained model disregards it and provides predictions with unwavering certainty.
/* **We take the ablation study in our main draft**, additionally, we use MMI benchmark to show the effectiveness of evidential loss.

|             | Text Retrieval    |      |      |      |         |           | Image Retrieval   |      |      |      |         |           |
|-------------|:-----------------:|------|------|------|---------|-----------|-------------------|------|------|------|---------|-----------|
|             | clean             |      | OOD  |      | OOD_avg | **MMI**       | clean             |      | OOD  |      | OOD_avg | **MMI**       |
|             | R@1               | R@1  | R@1  | R@1  | R@1     |           | R@1               | R@1  | R@1  | R@1  | R@1     |           |
| ELIP w/o A  | 60.2              | 51.7 | 49.8 | 43.1 | 48.2    | &darr;19.9%     | 44.5              | 38.4 | 36.1 | 30.6 | 35.0    | &darr;21.3%     |
| ELIP w/o IA | 71.3              | 62.1 | 63.8 | 55.1 | 60.3    | &darr;15.4%     | 52.8              | 45.6 | 44.3 | 38.1 | 42.7    | &darr;19.2%     |
| ELIP w/o TA | 76.6              | 63.8 | 68.0 | 55.6 | 62.5    | &darr;18.5%     | 60.1              | 51.0 | 51.5 | 42.3 | 48.3    | &darr;19.7%     |
| ELIP w/o Ev | 76.7              | 64.3 | 70.5 | 58.2 | 64.3    | &darr;16.1%     | 60.3              | 51.4 | 51.9 | 43.3 | 48.9    | &darr;19.0%     |
| ELIP        | 77.5              | 66.3 | 71.3 | 60.0 | 65.9    | &darr;**15.0%** | 60.4              | 51.5 | 52.3 | 43.7 | 49.2    | &darr;**18.6%** |

R2.4 The experiments are insufficient. The authors may want to compare the proposed model with baselines on more settings, such as changing the proportion of random noise during OOD generation.

A2.4 **Extra OOD evaluation:** We apprciated your review, I believe most reviews having the same concern about our limited evaluation. Thanks to reviewer 5LZY, we asses the effectiveness of our method in addressing a broader range of intricate out-of-distribution (OOD) cases, we adhered [1] and created various sets of perturbed images and texts.

In the first experiment, we selected three distinct **OOD-image** settings as highlighted in [1]. We randomly choose one perturbation from three categories (Blur: zoom, Weather: snow, Digital: JPEG), respectively. Our analysis encompasses results averaged across five perturbation levels. Presented below are the performance comparisons for Image-Text Retrieval between ELIP and baseline models on the **MS-COCO** dataset.

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

After testing on more complex image-OOD cases, we observe ELIP surpasses other baselines in most strong OOD settings except JPEG compression. Ffrom our observation, we found JPEG compressed image doesn't have a clear domain shift, therefore we assume this OOD setting follows the trend of image-text result on clean data, where BLIP-ft surpasses ELIP in ID case.

Experiment 2: We also randomly choose three **OOD-text** settings from three perturbation categories (character-level: keyboard, word-level: SR, sentence-level: formal) as mentioned in [1], all results are averaged on five perturbation levels. The following are Image-Text Retrieval performance comparision between ELIP and baseline models on **MS-COCO** dataset.

| **Image Retrieval** |          |              |          |          |          |          |      |            |      |
|---------------------|----------|--------------|----------|----------|----------|----------|------|------------|------|
|                     |          | **Keyboard** |          |          | **SR**   |          |      | **Formal** |      |
|                     | R@1      | R@5          | R@10     | R@1      | R@5      | R@10     | R@1  | R@5        | R@10 |
|     CLIP-zs         | 21.0     | 41.2         | 51.6     | 29.2     | 53.0     | 63.6     | 36.4 | 60.9       | 70.8 |
|     ALBEF-ft        | 38.0     | 63.4         | 73.0     | 52.4     | 77.7     | 85.5     | 60.2 | 83.9       | 90.3 |
|     BLIP-ft         | **42.7** | **67.5**     | **76.6** | 55.5     | 79.5     | 86.7     | **63.5** | **85.3**       | **91.2** |
|     ELIP            | 36.8     | 61.3         | 71.0     | **56.2** | **80.0** | **87.5** |  60.1    |   83.5         |  90.1    |

| **Text Retrieval** |          |              |          |          |          |          |          |            |          |
|--------------------|----------|--------------|----------|----------|----------|----------|----------|------------|----------|
|                    |          | **Keyboard** |          |          | **SR**   |          |          | **Formal** |          |
|                    | R@1      | R@5          | R@10     | R@1      | R@5      | R@10     | R@1      | R@5        | R@10     |
|     CLIP-zs        | 36.8     | 62.1         | 72.8     | 47.0     | 72.8     | 81.8     | 56.8     | 80.4       | 87.7     |
|     ALBEF-ft       | 57.9     | 82.6         | 89.6     | 70.1     | 90.6     | 95.1     | 77.6     | 94.1       | 97.0     |
|     BLIP-ft        | **64.1** | **86.4**     | **91.9** | **74.2** | **92.4** | **96.1** | **81.7** | **95.2**   | **97.6** |
|     ELIP           | 58.2     | 82.5         | 89.5     | 73.0     | 91.7     | 95.9     |   77.8       |    93.9        |    97.0      |

After testing on different type of text-OOD cases, we found ELIP doesn't perform as good as facing the image-OOD cases, but this follows the same trend in our own OOD settings (Table 1 main draft). We want to reemphasize this issue in two perspectives. 1) BLIP and ALBEF use cross-modal attention to align image and text during training. 2) BLIP and ALBEF have a two-stage evalution process, the image and text features were generated based on each other. Contrastively, ELIP is a retrieval-focus two-stream structure witout cross-modal attention, each modal has an independent encoder, which allows much faster inference time in retrieval task than BLIP/ ALBEF.  

Thanks to reviewer 5LZY, we can further our evluation using MMI benchmark. For OOD retrieval, we average over 5 Image-OOD (gaussian noise, random rotate, zoom-blur, snow, JPEG), 4 Text-OOD (natural noise, keyboard, SR, Formal) and 3 cross-OOD settings (mentioned in main draft). 

| **Image Retrieval** |      |           |      |      |             |      |           |           |          |
|---------------------|------|-----------|------|------|-------------|------|-----------|-----------|----------|
|                     |      | **Clean** |      |      | **OOD_avg** |      |           | **MMI**   |          |
|                     | R@1  | R@5       | R@10 | R@1  | R@5         | R@10 | R@1       | R@5       | R@10     |
|     CLIP-zs         | 35.3 | 60.0      | 70.2 | 28.4 | 51.6        | 62.2 | &darr;19.5%     | &darr;14.0%     | &darr;11.4%    |
|     ALBEF-ft        | 60.7 | 84.3      | 90.5 | 46.9 | 71.4        | 79.8 | &darr;22.8%     | &darr;15.4%     | &darr;11.8%    |
|     BLIP-ft         | 64.3 | 85.7      | 91.5 | 51.0 | 74.2        | 81.9 | &darr;20.8%     | &darr;13.4%     | &darr;10.5%    |
|     ELIP            | 60.4 | 83.9      | 90.5 | 50.1 | 74.9        | 83.3 | &darr;**17.1%** | &darr;**10.7%** | &darr;**8.0%** |

| **Text Retrieval** |      |           |      |      |             |      |           |          |          |
|--------------------|------|-----------|------|------|-------------|------|-----------|----------|----------|
|                    |      | **Clean** |      |      | **OOD_avg** |      |           | **MMI**  |          |
|                    | R@1  | R@5       | R@10 | R@1  | R@5         | R@10 | R@1       | R@5      | R@10     |
|     CLIP-zs        | 56.0 | 79.6      | 86.9 | 43.2 | 63.2        | 77.6 | &darr;22.9%     | &darr;20.7%    | &darr;10.8%    |
|     ALBEF-ft       | 77.6 | 94.3      | 97.2 | 60.8 | 81.5        | 87.3 | &darr;21.7%     | &darr;13.6%    | &darr;10.2%    |
|     BLIP-ft        | 81.9 | 95.4      | 97.8 | 70.7 | 84.2        | 89.2 | &darr;13.7%     | &darr;11.7%    | &darr;8.8%     |
|     ELIP           | 77.5 | 94.2      | 97.0 | 67.3 | 85.7        | 91.3 | &darr;**13.2%** | &darr;**9.0%** | &darr;**5.9%** |

We have observed that ELIP outperforms all other baseline models on the MMI benchmark, demonstrating the robustness of our method in diverse OOD settings. 

[1]: <https://arxiv.org/pdf/2212.08044.pdf> 
Qiu, Jielin et al. “Are Multimodal Models Robust to Image and Text Perturbations?” ArXiv abs/2212.08044 (2022): n. pag.

## Reviewer 3

We appreciate the Reviewer's approval and valuable comments. We respond to the Reviewer's concerns as below. Hope our explaination mitigate your concerns. 

R3.1 Could you explain how the Dirichlet distribution is used in the evidential deep learning framework to model uncertainty?

A3.1 **Dirichlet distribution and usage in ELIP:**, Dirichlet distribution is used to describe the sosine similarity distribution between a query sample and all target samples. In standard deep learning, uncertainty is often treated as a single scalar value, such as softmax probabilities or variance, which might not capture the complexity of uncertainty inherent in real-world scenarios. The evidential deep learning framework aims to address this limitation by representing uncertainty using the entire distribution over fusion probabilities.

R3.2 Can you expand on the role of Subjective Logic in quantifying uncertainty in this context?
A3.2 Subjective Logic is particularly useful in situations where there are multiple sources of information with varying levels of trustworthiness, or when dealing with subjective opinions and beliefs. ELIP mainly focus on image-text retrieval task, which involves feature alignment and ranking process that contains multiple sources aof information and different levels of trustworthiness, respectively. Therefore, we consider using Subjective Logic to quantify cross-modal retrieval uncertainty. 

R3.3 How is the belief mass calculated for each singleton in Subjective Logic, and what does it represent in the model's output?

A3.3 **Subjective logic and usage in ELIP:** Subjective Logic is a framework for reasoning under uncertainty that extends traditional probability theory to handle situations where incomplete or subjective information is available. In Subjective Logic, belief mass is used to quantify the degree of belief in different propositions or hypotheses. In our project, we use the cosine similarity between image an text to calculate the belief mass. Please follow Equation 3-5 in our paper for blief mass calculation.

R3.4 What is the significance of assigning the similarity vector ρ as the general representation for ρi2t and ρt2i?

A3.4 We use ρ as the general representation for ρi2t and ρt2i to enhance the clarity and consistency of our equation. Since ρi2t and ρt2i follow the same computation rule in Equation 6 and Equation 7 from our main draft. 

R3.5 In line 120, v_{cls} is not text embeddings, so it would be good to rephrase the sentence.

A3.5 In line 120, v_{cls} means iamge token.

R3.6 In line 189, "object" should be "objective".

A3.6 We will revise this in our final version.

## Reviewer 4
We appreciate the Reviewer's valuable comments and suggestions. We respond to the Reviewer's concerns as below.

R4.1 The OOD problem (Gaussian noise, image distortion, spelling errors) discussed in the article is not exclusive to image-text retrieval tasks but also exists in other tasks such as image captioning and image generation. Therefore, I think that the article's focus solely on image-text retrieval is insufficient.

A4.1 **Our Task & future plan:** ELIP and ELIP+ performs as encoder system and focuses on image-text alignment, where Image-text retrieval is a significant task to evaluate the cross-modal network's robustness. We didn't consider other tasks in this paper since this is our first exploration of robustness and uncertainty in cross-modal task, we believe image-text retrieval is sufficient to test out our ideas. In addition, our method can easily apply to other cross-modal tasks, we will discover ELIP in image-captioning, classification and QA tasks in the future work.

R4.2 The OOD settings discussed in the article are overly simplistic and may not adequately represent real-world scenarios. There seems to be a significant gap between the discussed settings and reality.

R4.3 There is already existing article [1]: <https://arxiv.org/pdf/2212.08044.pdf> that propose benchmarks for similar OOD problems, with more complex settings. I recommend that the authors follow the settings provided in [1] to evaluate the performance of their method.

A4.2 & A4.3 
** Extra OOD Evaluation:** We really thank for the Reviewer's pointing out this important related work. To validate our method's ability to handle more complex and diverse OOD cases, we followed [1] and generated different sets of perturbated images and texts.

In the first experiment, we selected three distinct **OOD-image** settings as highlighted in [1]. We randomly choose one kind of perturbation from three categories (Blur: zoom, Weather: snow, Digital: JPEG). Our analysis encompasses results averaged across five perturbation levels. Presented below are the performance comparisons for Image-Text Retrieval between ELIP and baseline models on the MS-COCO dataset.

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

After testing on more complex image-OOD cases, we observe ELIP surpasses other baselines in most strong OOD settings except JPEG compression. Ffrom our observation, we found JPEG compressed image doesn't have a clear domain shift, therefore we assume this OOD setting follows the trend of image-text result on clean data, where BLIP-ft surpasses ELIP in ID case.

Experiment 2: We also randomly choose three **OOD-text** settings from three perturbation categories (character-level: keyboard, word-level: SR, sentence-level: formal) as mentioned in [1], all results are average on five perturbation levels. The following are Image-Text Retrieval performance comparision between ELIP and baseline models on MS-COCO dataset.

| **Image Retrieval** |          |              |          |          |          |          |      |            |      |
|---------------------|----------|--------------|----------|----------|----------|----------|------|------------|------|
|                     |          | **Keyboard** |          |          | **SR**   |          |      | **Formal** |      |
|                     | R@1      | R@5          | R@10     | R@1      | R@5      | R@10     | R@1  | R@5        | R@10 |
|     CLIP-zs         | 21.0     | 41.2         | 51.6     | 29.2     | 53.0     | 63.6     | 36.4 | 60.9       | 70.8 |
|     ALBEF-ft        | 38.0     | 63.4         | 73.0     | 52.4     | 77.7     | 85.5     | 60.2 | 83.9       | 90.3 |
|     BLIP-ft         | **42.7** | **67.5**     | **76.6** | 55.5     | 79.5     | 86.7     | **63.5** | **85.3**       | **91.2** |
|     ELIP            | 36.8     | 61.3         | 71.0     | **56.2** | **80.0** | **87.5** |  60.1    |   83.5         |  90.1    |

| **Text Retrieval** |          |              |          |          |          |          |          |            |          |
|--------------------|----------|--------------|----------|----------|----------|----------|----------|------------|----------|
|                    |          | **Keyboard** |          |          | **SR**   |          |          | **Formal** |          |
|                    | R@1      | R@5          | R@10     | R@1      | R@5      | R@10     | R@1      | R@5        | R@10     |
|     CLIP-zs        | 36.8     | 62.1         | 72.8     | 47.0     | 72.8     | 81.8     | 56.8     | 80.4       | 87.7     |
|     ALBEF-ft       | 57.9     | 82.6         | 89.6     | 70.1     | 90.6     | 95.1     | 77.6     | 94.1       | 97.0     |
|     BLIP-ft        | **64.1** | **86.4**     | **91.9** | **74.2** | **92.4** | **96.1** | **81.7** | **95.2**   | **97.6** |
|     ELIP           | 58.2     | 82.5         | 89.5     | 73.0     | 91.7     | 95.9     |   77.8       |    93.9        |    97.0      |

After testing on different type of text-OOD cases, we found ELIP doesn't perform as good as facing the image-OOD cases, this follows the same trend in our own OOD settings (Table 1 main draft). We want to further explain this issue in three perspectives. 1) BLIP and ALBEF use cross-modal attention to align image and text during training. 2) BLIP and ALBEF have a two-stage evalution process, the image and text features were generated based on each other. ELIP is a two-stream structure witout cross-modal attention, and each modal has an independent encoder. 

Thanks to reviewer, we can further our evluation using MMI benchmark. For OOD retrieval, we average over 5 Image-OOD (gaussian noise, random rotate, zoom-blur, snow, JPEG), 4 Text-OOD (natural noise, keyboard, SR, Formal) and 3 cross-OOD settings (mentioned in main draft). 

| **Image Retrieval** |      |           |      |      |             |      |           |           |          |
|---------------------|------|-----------|------|------|-------------|------|-----------|-----------|----------|
|                     |      | **Clean** |      |      | **OOD_avg** |      |           | **MMI**   |          |
|                     | R@1  | R@5       | R@10 | R@1  | R@5         | R@10 | R@1       | R@5       | R@10     |
|     CLIP-zs         | 35.3 | 60.0      | 70.2 | 28.4 | 51.6        | 62.2 | &darr;19.5%     | &darr;14.0%     | &darr;11.4%    |
|     ALBEF-ft        | 60.7 | 84.3      | 90.5 | 46.9 | 71.4        | 79.8 | &darr;22.8%     | &darr;15.4%     | &darr;11.8%    |
|     BLIP-ft         | 64.3 | 85.7      | 91.5 | 51.0 | 74.2        | 81.9 | &darr;20.8%     | &darr;13.4%     | &darr;10.5%    |
|     ELIP            | 60.4 | 83.9      | 90.5 | 50.1 | 74.9        | 83.3 | &darr;**17.1%** | &darr;**10.7%** | &darr;**8.0%** |

| **Text Retrieval** |      |           |      |      |             |      |           |          |          |
|--------------------|------|-----------|------|------|-------------|------|-----------|----------|----------|
|                    |      | **Clean** |      |      | **OOD_avg** |      |           | **MMI**  |          |
|                    | R@1  | R@5       | R@10 | R@1  | R@5         | R@10 | R@1       | R@5      | R@10     |
|     CLIP-zs        | 56.0 | 79.6      | 86.9 | 43.2 | 63.2        | 77.6 | &darr;22.9%     | &darr;20.7%    | &darr;10.8%    |
|     ALBEF-ft       | 77.6 | 94.3      | 97.2 | 60.8 | 81.5        | 87.3 | &darr;21.7%     | &darr;13.6%    | &darr;10.2%    |
|     BLIP-ft        | 81.9 | 95.4      | 97.8 | 70.7 | 84.2        | 89.2 | &darr;13.7%     | &darr;11.7%    | &darr;8.8%     |
|     ELIP           | 77.5 | 94.2      | 97.0 | 67.3 | 85.7        | 91.3 | &darr;**13.2%** | &darr;**9.0%** | &darr;**5.9%** |

We have observed that ELIP outperforms all other baseline models on the MMI benchmark, demonstrating the effectiveness of our method in diverse OOD settings.


R4.4 Typo: Line 167 should be “Equation”.
A4.4 We will refine this in our final version.

## Reviewer 5

R5.1 What is the key component in improving the OOD robustness of VL models? Is it finetuning with adapters or introducing the evidential uncertainty?

A5.1 **Effectiveness evaluation:** We agree with the Reviewer's comment. However, we want to reemphasize the primary focus of this paper is to introduce an innovative approach for leveraging evidential learning in the image-text retrieval task, which enables improved uncertainty estimation. To achieve this, we employ adapters to efficiently fine-tune mainstream pre-trained Vision-and-Language (VL) models, such as CLIP and BLIP. Our method involves a gradual shift from a simple probability distribution (softMax) to a more robust Dirichlet Distribution (evidence) as the model's posterior, where the evidential loss played and absolute role in this process. Consequently, our model transforms into a deterministic uncertainty estimator.

/* **We take the ablation study in our main draft**, additionally, we use MMI benchmark to show the effectiveness of evidential loss. We hope this table can solve your concern.

|             | Text Retrieval    |      |      |      |         |           | Image Retrieval   |      |      |      |         |           |
|-------------|:-----------------:|------|------|------|---------|-----------|-------------------|------|------|------|---------|-----------|
|             | clean             |      | OOD  |      | OOD_avg | **MMI**       | clean             |      | OOD  |      | OOD_avg | **MMI**       |
|             | R@1               | R@1  | R@1  | R@1  | R@1     |           | R@1               | R@1  | R@1  | R@1  | R@1     |           |
| ELIP w/o A  | 60.2              | 51.7 | 49.8 | 43.1 | 48.2    | &darr;19.9%     | 44.5              | 38.4 | 36.1 | 30.6 | 35.0    | &darr;21.3%     |
| ELIP w/o IA | 71.3              | 62.1 | 63.8 | 55.1 | 60.3    | &darr;15.4%     | 52.8              | 45.6 | 44.3 | 38.1 | 42.7    | &darr;19.2%     |
| ELIP w/o TA | 76.6              | 63.8 | 68.0 | 55.6 | 62.5    | &darr;18.5%     | 60.1              | 51.0 | 51.5 | 42.3 | 48.3    | &darr;19.7%     |
| ELIP w/o Ev | 76.7              | 64.3 | 70.5 | 58.2 | 64.3    | &darr;16.1%     | 60.3              | 51.4 | 51.9 | 43.3 | 48.9    | &darr;19.0%     |
| ELIP        | 77.5              | 66.3 | 71.3 | 60.0 | 65.9    | &darr;**15.0%** | 60.4              | 51.5 | 52.3 | 43.7 | 49.2    | &darr;**18.6%** |

R5.2 Why does the proposed ELIP underperform in the OOD text?

A5.2 **Robustness evaluation:** We admit that ELIP underperform in the OOD text, but we want to reemphasize that the primary focus of our research is centered around two key objectives: achieving deterministic uncertainty estimation and enhancing model robustness in out-of-distribution (OOD) cases. In Table 1, our method, ELIP, exhibits remarkable performance, surpassing BLIP-ft in both OOD-image and OOD-image&text scenarios. This achievement is particularly noteworthy as we fine-tuned a significantly smaller subset of parameters (65M) compared to BLIP, which utilizes a more complex structure and fine-tunes the entire model. Instead of a direct retrieval accuracy comparison, we adopt a reference paper [1] suggested by reviewer 5LZy to ensure a more reasonable and objective evaluation. After reading, we found one benchmark named MultiModal Impact score, which can be used to analyze our existing experiment results, since **MMI** is used to easure the relative performance drop between the ID and OOD performance.


\* **In the following table, we grab the clean and average OOD retrieval resuls from Table 1(main draft) directly**. In addition, we provide the MMI comparision based on these results:

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

