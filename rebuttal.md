# Rebuttal

## Reviewer rBGR
`Q1`: Missing evaluation with GPT-4v.

`A1`: We thank the reviewer for this comment.
It is non-trivial to directly call GPT-4V API to do image-text retrieval, since it requires a proper prompt design. Also, we did not compare `ELIP` with the Multi-Modal LLM `GPT4V` in our project since ELIP (parameter size = 400M) is an encoder-based model , which is different from GPT4V (parameter size = 1.7T), a large generative model. 


## Reviewer G7DB
`Q1`: Since the ELIP+ is initialized by BLIP (line 569), why its performance is lower than BLIP in some settings (table 1). Could you provide a explanation?

`A1`: Thanks for the valuable comment.

We kindly summarize two explanations about the performance gap between ELIP+ and BLIP as follows:

*  BLIP has three loss terms: ITC, ITM, and LM. For ITC loss, they introduced a `momentum encoder` to generate soft labels, which improves vision-language understanding and leads to better model performance. However, the `momentum encoder` has millions of parameters and demand more GPU memory, which is limited by our computation resource.

* We follow BLIP’s implementation but simplify ELIP+’s architecture by eliminating all the momentum encoders, since we propose a light-weight fine-tuning method. By doing this, the `batch size` we use in ELIP+ (`200`) is still smaller than BLIP (`256`) used. As we observed, the batch size is essential in a contrastive learning framework, which can also explain why ELIP+ performs worse than BLIP in some cases.


## Reviewer J2CD
`Weakness`: Could experiment with wider range of real-world perturbations.
`R1`: Has the author tried other more perturbation methods to generate the OOD dataset?

`A1`: Thanks for the valuable comments.

We provided eight perturbations to simulate real-world noise cases in our main draft. To further test our method on a wider range of perturbation methods, we follow [1] by generating five more OOD cases based on MS-COCO.

* There are five perturbation levels of each OOD case, and we report the averaged Recall(R@k) of them in Table 1.



|			  |   	  |    	  | i2t      |    	  |    	  |    	  | t2i      |    	  |    	  |
|--------------|---------|----------|----------|----------|----------|----------|----------|----------|----------|
| Perturbation | Method  | R@1      | R@5      | R@10 	| Mean 	| R@1      | R@5      | R@10 	| Mean 	|
|			  | CLIP	| 42.4 	| 69.9 	| 79.9 	| 64.1 	| 34.9 	| 63.3 	| 74.9 	| 57.7 	|
| Shot   	  | ALBEF   | 66.2 	| 86.6 	| 92.0 	| 81.6 	| 52.1 	| 77.9 	| 85.8 	| 71.9 	|
|			  | BLIP	| 70.1 	| 88.2 	| 92.8 	| 83.7 	| 55.2 	| 79.2 	| 86.5 	| 73.7 	|
|			  | ELIP	| **71.8** | **90.1** | **94.4** | **85.5** | **55.7** | **80.2** | **87.7** | **74.6** |
|			  |   	  |    	  |    	  |    	  |    	  |    	  |    	  |    	  |    	  |
|			  | CLIP	| 35.6 	| 63.0 	| 74.3 	| 57.6 	| 29.8 	| 58.3 	| 70.7 	| 53.0 	|
| Impulse      | ALBEF   | 66.0 	| 86.8 	| 92.1 	| 81.6 	| 52.1 	| 77.9 	| 85.8 	| 71.9 	|
|			  | BLIP	| 68.7 	| 87.6 	| 92.3 	| 82.9 	| 54.5 	| 78.6 	| 86.1 	| 73.1 	|
|			  | ELIP	| **72.3** | **90.4** | **94.7** | **85.8** | **56.7** | **81.1** | **88.5** | **75.4** |
|			  |   	  |    	  |    	  |    	  |    	  |    	  |    	  |    	  |    	  |
|			  | CLIP	| 43.7 	| 71.7 	| 81.5 	| 65.6 	| 35.2 	| 63.8 	| 75.2 	| 58.1 	|
| Defocus      | ALBEF   | 62.6 	| 84.1 	| 90.1 	| 79.0 	| 50.6 	| 75.7 	| 83.9 	| 70.1 	|
|			  | BLIP	| 68.0 	| 87.5 	| 92.2 	| 82.6 	| 54.6 	| 78.3 	| 85.4 	| 72.8 	|
|			  | ELIP	| **68.3** | **89.1** | **94.2** | **83.9** | **56.0** | **80.4** | **88.0** | **74.8** |
|			  |   	  |    	  |    	  |    	  |    	  |    	  |    	  |    	  |    	  |
|			  | CLIP	| 36.5 	| 65.7 	| 77.1 	| 59.8 	| 36.5 	| 65.7 	| 77.1 	| 59.8 	|
| Speckle      | ALBEF   | 69.9 	| 89.3 	| 94.1 	| 84.4 	| 54.7 	| 80.1 	| 87.6 	| 74.1 	|
|			  | BLIP	| **74.4** | **91.5** | 95.0 	| **87.0** | **58.4** | **81.6** | **88.5** | **76.2** |
|			  | ELIP	| 73.1 	| 91.0 	| 95.1 	| 86.4 	| 56.6 	| 81.0 	| 88.3 	| 75.3 	|
|			  |   	  |    	  |    	  |    	  |    	  |    	  |    	  |    	  |    	  |
|			  | CLIP	| 32.4 	| 58.3 	| 68.9 	| 53.2 	| 27.3 	| 53.8 	| 65.7 	| 48.9 	|
| Pixel  	  | ALBEF   | 45.9 	| 65.7 	| 72.7 	| 61.4 	| 36.3 	| 58.9 	| 67.5 	| 54.2 	|
|			  | BLIP	| 56.1 	| 76.3 	| 82.6 	| 71.6 	| 44.9 	| 68.3 	| 76.5 	| 63.3 	|
|			  | ELIP	| **67.1** | **88.6** | **93.4** | **83.0** | **54.8** | **79.1** | **86.9** | **73.6** |

_Table 1. Comparison of performance in terms of Recall@K (R@K) among OOD retrieval. CLIP ZS is pretrained zero-shot, all other models are fine-tuned on MS-COCO._

From the new OOD retrieval experiments, we observed that ELIP surpasses other methods in most cases.



## Reviewer jcV3
Novelty:

`Weakness1`: It’s an integration of existing modules and the overall novelty of this framework is marginal.

`W_A1`: We found deep evidential is a well-studied robust prediction method in the single-domain classification and regression tasks, and such method is proved to be useful for uncertainty estimation. However, our paper focuses on multi-modal understanding and ranking tasks, where the research of using deep evidential to improve the robustness of multi-modal embedding is insufficient.

We kindly summarize the technical contributions of this work as follows:
<ol>
<li>We introduce a method that can be plugged into most of the transformer-based multi-modal learning framework (e.g., CLIP, BLIP).</li>
<li>Our proposed method improves the robustness of pre-trained models when facing OOD cases.</li>
<li>We improve the efficiency of fine-tuning a robust-prediction vision-language model, achieving performance boost with much shorter training time and lower computational cost.</li>
</ol>
`Q1`: How does the proposed evidential learning approach compare to existing methods in terms of performance, robustness, and efficiency?

`A1`: We appreciate this valuable suggestion.
As the reviewer `jcV3` suggested, the deep ensemble method is another solid approach for robust model prediction. Therefore, to directly compare our method with the deep ensemble approach, we provide a randomization-based adapter ensemble, since the whole model ensemble demands high computational cost and training time. Table 1 provides a comparison result of deep ensemble and ELIP among ID and OOD retrieval based-on MS-COCO. 



|       |               |              |      i2t     |              |              |      t2i     |              |
|-------|---------------|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
| Noise | Method        |      R@1     |      R@5     |     R@10     |      R@1     |      R@5     |     R@10     |
| None  | Ensemble(n=3) | 76.0$\pm$0.5 | 92.9$\pm$0.2 | 96.5$\pm$0.2 | 58.5$\pm$0.6 | 82.7$\pm$0.2 | 88.8$\pm$1.0 |
|       | Ensemble(n=5) |              |              |              |              |              |              |
|       | ELIP          |   **78.4**   |   **93.6**   |   **97.0**   |   **60.4**   |   **83.5**   |   **90.2**   |
| Image | Ensemble(n=3) | 65.1$\pm$0.4 | 85.8$\pm$0.2 | 91.8$\pm$0.3 | 50.3$\pm$0.2 | 76.1$\pm$0.4 | 84.6$\pm$0.1 |
|       | Ensemble(n=5) |              |              |              |              |              |              |
|       | ELIP          |   **67.2**   |   **86.4**   |   **92.0**   |   **51.9**   |   **76.7**   |   **85.1**   |
| Text  | Ensemble(n=3) | 69.1$\pm$0.3 | 89.4$\pm$0.1 | 94.2$\pm$0.1 | 50.3$\pm$0.7 | 75.7$\pm$0.4 | 84.3$\pm$0.3 |
|       | Ensemble(n=5) |              |              |              |              |              |              |
|       | ELIP          |   **72.0**   |   **90.6**   |   **94.8**   |   **52.3**   |   **77.0**   |   **85.0**   |
| Cross | Ensemble(n=3) | 58.8$\pm$0.2 | 81.7$\pm$0.1 | 88.7$\pm$0.2 | 42.8$\pm$0.2 | 68.8$\pm$0.4 | 78.3$\pm$0.2 |
|       | Ensemble(n=5) |              |              |              |              |              |              |
|       | ELIP          |   **59.7**   |   **82.7**   |   **89.4**   |   **44.5**   |   **70.0**   |   **79.2**   |

_Table 1: We train each model in parallel for 10 epochs with batch size as 280. The deep ensemble involves `n` models with different adapter's initialization, and we take the averaged Recall of all n moldes._


`Q2`: Are there clear advantages or limitations identified in the comparison? I would like to see more discussions and comparisons in the experimental session.

`A2`:We appreciate this valuable comment.
Based on the experiments, we provide the following analysis.

**Pros**
* ELIP can provide uncertainty estimation and retrieval results in a single forward process.

**Cons**
* Deep ensemble requires longer training time, inference time, and computation cost. Suppose the NNs in the deep ensemble are the same as ELIP, and the number of NNs in the ensemble is M. With the same training and test time $T_{train}$, $T_{test}$ for each NN, deep ensemble will spend $M * (T_{train} + T_{test})$, and ELIP only requires $T_{train} + T_{test}$. Specifically, in our project, $T_{train} =28h $  when distributed training one NN for 10 epochs on 2*GPU (40GB), and $T_{test}=3 min$.  
 


`Q3`: Since some noises are manually simulated, to what extent does the proposed method generalize to diverse datasets and real-world scenarios?

`A3`: Thanks for the useful comments!

To test the generalization of ELIP, we generated five more OOD cases based on MS-COCO and provided the comparison results below.

There are five perturbation levels of each OOD case, and we report the averaged Recall(R@k) of them in Table 2.

|			  |   	  |    	  | i2t      |    	  |    	  |    	  | t2i      |    	  |    	  |
|--------------|---------|----------|----------|----------|----------|----------|----------|----------|----------|
| Perturbation | Method  | R@1      | R@5      | R@10 	| Mean 	| R@1      | R@5      | R@10 	| Mean 	|
|			  | CLIP	| 42.4 	| 69.9 	| 79.9 	| 64.1 	| 34.9 	| 63.3 	| 74.9 	| 57.7 	|
| Shot   	  | ALBEF   | 66.2 	| 86.6 	| 92.0 	| 81.6 	| 52.1 	| 77.9 	| 85.8 	| 71.9 	|
|			  | BLIP	| 70.1 	| 88.2 	| 92.8 	| 83.7 	| 55.2 	| 79.2 	| 86.5 	| 73.7 	|
|			  | ELIP	| **71.8** | **90.1** | **94.4** | **85.5** | **55.7** | **80.2** | **87.7** | **74.6** |
|			  |   	  |    	  |    	  |    	  |    	  |    	  |    	  |    	  |    	  |
|			  | CLIP	| 35.6 	| 63.0 	| 74.3 	| 57.6 	| 29.8 	| 58.3 	| 70.7 	| 53.0 	|
| Impulse      | ALBEF   | 66.0 	| 86.8 	| 92.1 	| 81.6 	| 52.1 	| 77.9 	| 85.8 	| 71.9 	|
|			  | BLIP	| 68.7 	| 87.6 	| 92.3 	| 82.9 	| 54.5 	| 78.6 	| 86.1 	| 73.1 	|
|			  | ELIP	| **72.3** | **90.4** | **94.7** | **85.8** | **56.7** | **81.1** | **88.5** | **75.4** |
|			  |   	  |    	  |    	  |    	  |    	  |    	  |    	  |    	  |    	  |
|			  | CLIP	| 43.7 	| 71.7 	| 81.5 	| 65.6 	| 35.2 	| 63.8 	| 75.2 	| 58.1 	|
| Defocus      | ALBEF   | 62.6 	| 84.1 	| 90.1 	| 79.0 	| 50.6 	| 75.7 	| 83.9 	| 70.1 	|
|			  | BLIP	| 68.0 	| 87.5 	| 92.2 	| 82.6 	| 54.6 	| 78.3 	| 85.4 	| 72.8 	|
|			  | ELIP	| **68.3** | **89.1** | **94.2** | **83.9** | **56.0** | **80.4** | **88.0** | **74.8** |
|			  |   	  |    	  |    	  |    	  |    	  |    	  |    	  |    	  |    	  |
|			  | CLIP	| 36.5 	| 65.7 	| 77.1 	| 59.8 	| 36.5 	| 65.7 	| 77.1 	| 59.8 	|
| Speckle      | ALBEF   | 69.9 	| 89.3 	| 94.1 	| 84.4 	| 54.7 	| 80.1 	| 87.6 	| 74.1 	|
|			  | BLIP	| **74.4** | **91.5** | 95.0 	| **87.0** | **58.4** | **81.6** | **88.5** | **76.2** |
|			  | ELIP	| 73.1 	| 91.0 	| 95.1 	| 86.4 	| 56.6 	| 81.0 	| 88.3 	| 75.3 	|
|			  |   	  |    	  |    	  |    	  |    	  |    	  |    	  |    	  |    	  |
|			  | CLIP	| 32.4 	| 58.3 	| 68.9 	| 53.2 	| 27.3 	| 53.8 	| 65.7 	| 48.9 	|
| Pixel  	  | ALBEF   | 45.9 	| 65.7 	| 72.7 	| 61.4 	| 36.3 	| 58.9 	| 67.5 	| 54.2 	|
|			  | BLIP	| 56.1 	| 76.3 	| 82.6 	| 71.6 	| 44.9 	| 68.3 	| 76.5 	| 63.3 	|
|			  | ELIP	| **67.1** | **88.6** | **93.4** | **83.0** | **54.8** | **79.1** | **86.9** | **73.6** |

_Table 2. Comparison of performance in terms of Recall@K (R@K) among OOD retrieval. CLIP ZS is pretrained zero-shot, all other models are fine-tuned on MS-COCO._

`Q4`: Are there insights into the model's transferability across different tasks and domains?

`A4`: Thanks for the valuable suggestion!
To test the transferability across domains, Table 3 provides comparison results of zero-shot retrieval on Flickr30K. Here we use the clean Flickr30k as the target domain test-set, and all methods are fine-tuned on the source domain MS-COCO. Besides this, we also provide zero-shot performance of CLIP (pre-training), which is not fine-tuned on MS-COCO. 

|    		 | 		 | i2t 	 |		 | 	 | t2i 	 | 		 |
|-------------|----------|----------|---------|------|----------|----------|
| Method 	 | R@1 	 | R@5 	 | R@10    | R@1  | R@5 	 | R@10     |
| CLIP  	 | 88.0    | 98.7     | 99.4   | 68.7 | 90.6     | 95.2     |
| ALBEF$^\dagger$  	 | 94.1     | 99.5     | 99.7    | 82.8 | 96.3     | 98.1     |
| BLIP$^\dagger$   	 | 94.8     | **99.7** | **100** | 84.9 | 96.7     | 98.3     |
| ELIP w/o EV | 93.4     | 99.3     | 99.7    | 82.3 | 96.2     | 98.2     |
| ELIP   	 | **95.2** | 99.6     | 99.9    | 83.9 | **97.1** | **98.6** |

_Table 3: Comparison of zero-shot image-text retrieval on Flickr30k. For ALBEF$^\dagger$ and BLIP$^\dagger$, we collect the results directly from [1]._

Since image captioning is out of the page of this project, we will leave this study for future work.

`Q5`: Given the computational demands associated with some probabilistic approaches, how does the proposed evidential learning method address issues related to scalability and efficiency?

`A5`:  With our lightweight robust fine-tuning method, we can scale it to most of the transformer-based multi-modal model that are trained using contrastive loss.
As for the efficiency, ELIP only fine-tunes the extra adapters. Therefore, ELIP is more efficient than the deep ensemble method, which usually trains multiple models and takes extra steps to perform the inference.

[1]: 1.Li, J., Li, D., Xiong, C. & Hoi, S. BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation.



## Reviewer 6jGU

`Weakness` : ELIP+ improvement over ELIP is never discussed in detail and simply just appears in page 6.

`A-W`: From our experiments, we found ELIP+ does not improve BLIP as much as ELIP improves CLIP. We kindly summarize two potential reasons for this as follows:

* BLIP has three loss terms: ITC, ITM, and LM. For ITC loss, they introduced a `momentum encoder` to generate soft labels, which improves vision-language understanding and leads to better model performance. However, the `momentum encoder` has millions of parameters and demand more GPU memory, which is limited by our computation resource.

* We follow BLIP’s implementation but simplify ELIP+’s architecture by eliminating all the momentum encoders, since we propose a light-weight fine-tuning method. By doing this, the `batch size` we use in ELIP+ (`200`) is still smaller than BLIP (`256`) used. As we observed, the batch size is essential in a contrastive learning framework, which can also explain why ELIP+ performs worse than BLIP in some cases.

We observed that our method has a larger improvement on CLIP than BLIP. One explanation is, we believe the pre-trained CLIP has better generalization than BLIP. Specifically, CLIP pre-trained on a web-scale dataset, but BLIP pre-trained on COCO and many other datasets.
This paper is mainly focused on ELIP, since ELIP is much faster to train and test than ELIP+. 

`Q1`: What is the explanation on the seeming mis-match of MMI computations?

A1: We really appreciate the meticulous comments.

We kindly make a clarification as follows. 

In this project, we compute $MMI = (R@k_{clean}-R@k_{ood})/R@k_{clean}$. In the main draft (Table1 & Table3), we compute $R@k_{ood}=(R@k_{text-ood}+R@k_{image-ood}+R@k_{cross-ood})/3$.

It is our negligence that we didn’t provide a detailed explanation about how we compute the MMI in the main draft, and we will add this computing process in our final version.

`Q2`: Why use MMI in the first place?

`A2`: Thanks for the valuable insights.

* Intuitively, one of the key reasons we use MMI in our project is because we think MMI not only describes the impact of one perturbation on the model's performance [1] but also presents the robustness of the model. Also, analyzing the performance drop (MMI) between ID and OOD retrieval can support the effectiveness of our method, since we mainly focus on how the model performs when facing OOD cases.

* After research, we found `RSUM` proposed in [2] can be another metric to evaluate the model's robustness. Where RSUM = SUM(i2t(R@1,R@5,R@10)+t2i(R@1,R@5,R@10)). Therefore, besides the old perturbations, we additionaly generate six new perturbations (Keyboard, shot, impulse, speckle, defocus, pixel) based on `MS-COCO`, and present our analysis in Table 1.

| Method   | Clean |AVG OOD| Shot  | Impulse | Speckle | Defocus | Pixel | Zoom| Snow| JPEG| Keyboard| SR| Formal| MMI |
|----------|-------|-------|---------|---------|---------|-------|---------|-----|-----|-----|-----|-----|-----|-----|
| CLIP ZS  | 394.5 | 339.1| 361.2 | 330.2   | 368.7   | 358.7   | 308.2 |  294.6|294.7|388.0|285.5|347.5|393.0|14.0%|
| CLIP  | 420.5 | 349.8|365.3 | 331.7   | 381.5   | 371.0   | 306.4 |291.0|289.3|402.1|316.1|376.2|417.3|16.8%|
| ALBEF | 504.6 |422.0| 460.6 | 460.3   | 376.4   | 447.1   | 347.0 | 282.2|408.8|480.9|404.5|471.4|503.1  |16.4%|
| BLIP  | 516.6 | 450.2|472.1 | 467.7   | **489.5**   | 466.1   | 404.7 | 291.6|432.8|**499.6**|**429.1**|**484.3**|**514.4** |12.9%|
| ELIP 	| 503.5 |**463.1**| **480.0** | **483.7**   | 485.0   |  **476.2**  | **469.8** |**368.6**| **448.3**|496.9|399.3|**484.3**|502.4|8.0%|

_Table 1: Comparison of performance in terms of RSUM and MMI among OOD retrieval. Where CLIP ZS is the pre-trained zero-shot evaluation, all other methods are fine-tuned on MS-COCO. `ave` is the average RSUM of all OOD retrieval._

From our observations, even ELIP has relatively lower RSUM on ID retrieval, but it presents higher RSUM in most OOD cases, which indicates the robustness of ELIP when facing noisy images and text in retrieval tasks. Also, it is predictable that BLIP has better performance when dealing with some text OOD cases, since they put more efforts on improving language understanding.

`Q3`: What is ELIP+?
`A3`: We kindly provide a clarification here. ELIP+ is our method plugged into BLIP, and ELIP is our method plugged into CLIP. Where BLIP and CLIP are two well-known vision-language modeling frameworks.

`Q4`: What dataset was used for the ablation study?

`A4`: Our ablation study is mainly focused on MS-COCO, since MS-COCO provides more distinct phenomena than Flickr30k. We use Gaussian Noise and Natural Noise as the image and text perturbation to generate test data for OOD retrieval. 

`Q5`: What are the parameters used for the noise-adding processes?

`A5`:
This paper applies various noise-adding techniques.

<ol>
<li>For the simple OOD image generation, we use the following parameters.</li>
 <ol>
<li>Gaussian: $\sigma=0.1$, $\mu=0$ 	 </li>
<li>Rotation: random(0$^\circ$,180$^ \circ$) </li>
 </ol>
 
<li>For other image perturbations, we use the following parameters.Suppose we have an image X, and the generated noisy image as X’. We list the generation process of a few perturbations below.</li>
<ol>
<li> The variables are used to control the noisy level (1-5), 5 means the noisiest.</li>
<li>Snow: X’ = X + snow_layer(loc, scale, clip, radius, sigma)</li>
<ol>
 Snow layer:
	<li>Level1: (0.1,0.3,3,0.5,10,4,0.8)</li>
	<li>Level2: (0.2,0.3,2,0.5,12,4,0.7)</li>
	<li>Level3: (0.55,0.3,4,0.9,12,8,0.7)</li>
	<li>Level4: (0.55,0.3,4.5,0.85,12,8,0.65)</li>
	<li>Level5: (0.55,0.3,2.5,0.85,12,12,0.55)</li>
</ol>	
<li>Zoom: X’ = (X + zoom(zoom factors)) / #zoom factors
<ol>
	Zoom factors:
<li>Level1: [1, 1.01, 1.02 … 1.11]</li>
<li>Level2: [1, 1.01, 1.02, … 1.16]</li>
<li>Level3: [1, 1.02, 1.04, … 1.21]</li>
<li>Level4: [1, 1.02, 1.04, … 1.26]</li>
<li>Level5: [1, 1.03, 1.06, … 1.33]</li>
</ol>
<li>Defocus: X’ = defocus(X, kernel(radius, alias_blur))</li>
<ol>
	Kernel:
	<li>Level1: (3, 0.1)</li>
	<li>Level2: (4, 0.5)</li>
	<li>Level3: (6, 0.5)</li>
	<li>Level4: (8, 0.5)</li>
	<li>Level5: (10, 0.5)</li>
</ol>
</ol>
<li>For the text perturbations, we use the following parameters. Suppose we have a caption X, and the generated noisy caption as X’.</li>
<ol>
	<li>Natural Noise: X’ = casing(diacritics(punctuation(spelling(whitespace(word-order(wrong suffix/prefix(X))))))). Natural noise is a mixture of different noisy aspects, and we sample the error rate of each aspect from a random distribution. We can control the overall noisy value by setting the mean of the random distribution. In this project, we use 3 as the mean. </li> 
</ol>
</ol>


We are incapable of listing all parameters for the natural noise text [3] and the other perturbations [1], since each of them requires a complex generation process.

[1]: Qiu, J., Zhu, Y., Shi, X., Wenzel, F., Tang, Z., Zhao, D., Li, B., & Li, M. (2022). Are Multimodal Models Robust to Image and Text Perturbations? ArXiv, abs/2212.08044.
[2]: Wu, H., Mao, J., Zhang, Y., Jiang, Y., Li, L., Sun, W., & Ma, W. (2019). Unified Visual-Semantic Embeddings: Bridging Vision and Language With Structured Meaning Representations. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 6602-6611.
[3]: Jakub N’aplava, Martin Popel, Milan Straka, and Jana Strakov’a. 2021. Under-
standing Model Robustness to User-generated Noisy Texts. ArXiv abs/2110.07428
(2021).















