---
layout: post
title: '[Paper Review] Unsupervised Deep Embedding for Clustering Analysis'
description: >
  거꾸로 읽는 self-supervised learning의 첫 번째 논문
subtitle: Deep Enbedded Clustering
date: '2022-06-09'
categories:
    - study
tags:
    - DEC
comments: true
published: true
last_modified_at: '2022-06-12'
---

본 논문은 2016년 PMLR에 실렸으며 feature representations과 cluster assignment를 동시에 학습하는 Deep Embedded Clustering(DEC)을 제안하였습니다. 설명에 앞서 슈퍼짱짱님의 블로그를 참고하였음을 밝힙니다.

- Table of Contents
{:toc .large-only}


## Introduction

***

클러스터링(Clustering)은 데이터 분석 및 시각화(visualization)에서 핵심적인 기법으로, 각기 다른 관점에서 unsupervised machine learning으로 널리 연구되어 왔습니다.

클러스터링 알고리즘에서 distance(또는 dissimilarity)는 중요하다고 하는데요. Distance는 feature space에서 데이터를 표현하는데 중요한 역할을 하기 때문이라고 이를 설명하였습니다. 예를 들어, k-means 클러스터링 알고리즘에서는 feature space에서 points 사이의 Euclidean distance를 사용하였습니다.

또한, feature space를 선택하는 것도 중요합니다. 가장 간단한 이미지 데이터셋을 제외하고는, raw pixels에서 Euclidian distance를 사용하는 것은 비효율적입니다.

결국 저자들은 다음과 같은 의문에 도달하였습니다. ""데이터 기반 접근 방식으로 feature space와 cluster memberships를 동시에 해결할 수 없을까?""

본 연구에서는 현재의 soft cluster assignment에서 도출된 보조 타겟 분포(auxiliary target distribution)을 사용하여 clusters를 재정의하는 방법을 제안하였습니다. 이를 통해 클러스터링뿐만 아니라 feautre representation도 개선시켰습니다. 이 실험은 이미지와 텍스트 데이터셋에서 정확도와 running time 모두 최신의 클러스터링 기법들보다 향상된 성능을 보였다고 합니다. 게다가 DEC는 hyperparameters 선택에 있어써도 훨씬 덜 민감했습니다.


### Contributions

***

* Deep embedding과 clustering의 공동 최적화
* Soft assignment를 통한 clusters 재정의
* 정확도 및 속도에서 SOTA clustering 달성


## Deep embeddded clustering

***

클러스터링을 data space X에서 바로 하는 것 대신에, 본 논문은 먼저 non-linear mapping $f_θ$로 data space X에 있는 data를 latent space Z로 변환하였습니다. Z의 차원은 "curse of dimensionality"를 피하기 위해 X 보다 작아야 했습니다. 본 연구에서 제안하는 알고리즘 DEC는 feature space Z에서 cluter center {$μ_j \in Z$}$_{j=1}^k$를 학습하고, data를 Z로 mapping하는 DNN의 파라미터 θ를 학습하면서 동시에 데이터를 클러스터링 하였습니다.

Deep embedded clustering (DEC)는 두 단계로 이루어져 있습니다.    
1. Parameter initialization with a deep autoencoder   
2. Parameter optimization (i.e., clustering)   
    * 보조 타겟 분포(auxiliary target distribution)를 계산하고 Kullback-Leibler (KL) divergence를 최소화하는 것을 반복하여 최적화합니다.

논문의 순서는 다음으로 "Clustering with KL divergence"가 나오지만 원활한 설명을 위하여 "Parameter initialization" 먼저 작성하겠습니다.
{:.message}

### Parameter initialization

***

DNN parameters $θ$와 cluster centroids {$μ_j$}를 초기화하는(initialize) 방법을 알아보겠습니다.

DEC network의 $θ$를 초기화하기 위하여 **Stacked autoencoder(SAE)**가 활용되었습니다. SAE의 각 레이어는 random corruption 이후 이전 계층의 츨력을 재구성하도록 학습된 denoising autoencoder로 초기화되었습니다. Denoising autoencoder는 다음과 같이 2개의 layer로 이루어져 있습니다.

$$
  \tilde{x} \sim Dropout(x) \\[0.5em]
  h = g_1(W_1\tilde{x} + b_1) \\[0.5em]
  \tilde{h} \sim Dropout(h) \\[0.5em]
  y = g_2(W_2\tilde{h} + b_2) \\[0.5em]
$$

Stacked autoecoder는 여러 개의 히든 레이어를 가지는 오토인코더이며, 레이어를 추가할수록 오토인코더가 더 복잡한 코딩(부호화)을 학습할 수 있게 됩니다. Denoising autoencoder는 입력에 noise를 추가하고 noise가 없는 원본 입력을 재구성하도록 학습시는 방법입니다. Stacked autoencoder 및 denoising autoencoder를 포함하여 autoencoder에 대한 자세한 설명은 Excelsior-JH님의 [오토인코더 (AutoEncoder)](#https://excelsior-cjh.tistory.com/187)를 참고하시길 바랍니다.
{:.message}

<br>

학습은 least squares loss $||x-y||^2$을 최소화함으로써 이루어집니다. 하나의 layer를 학습한 후, 다음 layer를 학습하기 위해 output $h$를 input으로 사용합니다. . 이러한 greedy lyaer-wise training 이후, reverse layer-wise training 순서로 모든 decoder layers 뒤에 모든 encoder layers를 붙여서 deep autoencoder를 형성하고 다음으로 재구성 손실(reconstruction loss)를 최소화하도록 학습합니다. 최종적으로 중간에 bottleneck coding layer가 있는 multilayer deep autoencoder가 됩니다.

![Network structure](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/paper_review/2022-06-09-DEC/network_structure.PNG?raw=true){:.aligncenter}<center><span style="color:gray; font-size:80%">Network structure</span></center>   
<br>

다음으로 상단의 그림과 같이 decoder layer를 버리고 econdoer layers를 data space와 feature space 간의 initial mapping으로 사용합니다. Cluster centers를 초기화하기 위해 데이터를 초기화된 DNN을 거쳐 embedded data를 얻은 다음 feature space Z에서 k-means clustering하여 사용하여 $k$개의 initial centroids $\lbraceμ_j\rbrace_{j=1}^k$를 얻습니다.

### Parameter optimization

***

Non-linear mapping $f_θ$과 cluster centroids {$μ_j$}$_{j=1}^k$의 초기값을 추정하였으므로, 비지도 알고리즘을 사용하여 clustering을 개선하는 방법을 살펴보겠습니다.

#### Clustering with KL divergence

***

KL divergence 기반 clustering은 다음의 두 단계를 반복하여 이루어집니다.   

**Step 1.** X → Z로 mapping된 embedded points와 cluster centroids 간의 **soft assignment를 계산**합니다.   
⇒ Embedded points와 cluster centroids 간의 거리를 계산하여, Embedded point가 cluster에 속할 확률(soft assignment)를 구하는 것입니다.   

**Step 2.** Deep mapping $f_θ$을 업데이트하고 보조 타겟 분포(auxiliary target distribution)를 통해 높은 신뢰도(high confidence)로 학습하여 cluster centroids를 재정의합니다.    
⇒ **보조 타겟 분포를 label로 사용**함으로써, unsupervised learning 알고리즘인 클러스터링이 마치 supverised learning 처럼 학습되어 높은 신뢰도로 학습한다고 말할 수 있습니다.{:.message}

이 절차를 수렴 기준에 충족될 때까지 반복합니다.

#### SOFT ASSIGNMENT

***

Embedded points $z_i$와 cluster centroids $μ_j$ 간의 유사도를 구하기 위하 t-분포(Studetnt's t-distribution)를 사용하였습니다. 

$$
  q_{ij} = \frac{(1+||z_i - μ_j||^2 / α)^- \frac{α+1}{2}}{\sum_{j'}(1+||z_i - μ_j||^2 / α)^- \frac{α+1}{2}}
$$

α는 t-분포의 자유도(degree of freedom)를 나타내며, **$q_{ij}$는 sample $i$가 cluster $j$에 속할 확률(i.e., soft assignment)**을 나타냅니다. Clustering은 비지도 알고리즘으로써 alpha를 validation set에 cross-validate하지 못하므로 모든 실험에서 alpha를 1로 설정하였습니다.

<$q_{ij}$는 어떻게 도출되었을까?>

t-분포의 공식은 다음과 같습니다.

$$
  f(t) = \frac{Γ(\frac{α+1}{2})}{\sqrt{απ}Γ(\frac{α}{2})}(1+\frac{t^2}{α})^{-\frac{α+1}{2}}
$$


t-분포를 논문에 맞게 적용해보자면, 데이터 $t$는 두 점 사이의 거리 $||z_i - μ_j||$가 되며 식은 다음과 같이 정리됩니다.
$$
\begin{aligned}
  q_{ij} &= \frac{\frac{Γ(\frac{α+1}{2})}{\sqrt{απ}Γ(\frac{α}{2})}(1+\frac{||z_i - μ_j||^2}{α})^{-\frac{α+1}{2}}}{\sum_{j'}\frac{Γ(\frac{α+1}{2})}{\sqrt{απ}Γ(\frac{α}{2})}(1+\frac{||z_i - μ_j||^2}{α})^{-\frac{α+1}{2}}} \\[2em]
         &= \frac{\frac{Γ(\frac{α+1}{2})}{\sqrt{απ}Γ(\frac{α}{2})}(1+\frac{||z_i - μ_j||^2}{α})^{-\frac{α+1}{2}}}{\frac{Γ(\frac{α+1}{2})}{\sqrt{απ}Γ(\frac{α}{2})}\sum_{j'}(1+\frac{||z_i - μ_j||^2}{α})^{-\frac{α+1}{2}}} \\[2em]
         &= \frac{(1+\frac{||z_i - μ_j||^2}{α})^{-\frac{α+1}{2}}}{\sum_{j'}(1+\frac{||z_i - μ_j||^2}{α})^{-\frac{α+1}{2}}} \\[2em]
\end{aligned}
$$
{:.message}

#### KL DIVERGENCE MINIMIZATION

***

다음으로 저자들은 보조 타겟 분포(auxiliary target distribution)를 통해 높은 신뢰도(high confidence)로 학습하면서 clusters를 재정의하였습니다. 

기존의 clustering은 unsupervised learning으로 사용되었지만, 본 논문에서는 보조 타겟 분포를 label로 사용하여 마치 supervised learning 처럼 학습하였으므로 높은 신뢰도(high confidence)로 clusters를 재정의했다고 할 수 있습니다.{:.message}

구체적으로는 DEC는 soft assignments를 target distribution에 매칭하면서 학습합니다. 끝으로, soft assignments $q_{ij}$와 target distribution $p_{ij}$ 간의 KL divergence loss가 목적함수로 정의되었습니다.

$$
  L = KL(P||Q) = \sum_i\sum_jp_{ij}\log\frac{p_{ij}}{q_{ij}}
$$


##### (참고) KL DIVERGENCE에 대한 설명
KL divergence(Kullback-Leibler divergence)는 **두 확률분포의 차이를 계산**하는데에 사용되는 함수입니다. 두 확률변수에 대한 확률분포 $P$, $Q$가 있을 때, 두 분포의 KL divergence는 다음과 같이 정의할 수 있습니다.

$$D_{KL}(P||Q) = \sum_i P(i)\log \frac{P(i)}{Q(i)}$$

텐서플로우 공식 문서에 정의되어 있는 용어로 설명해보자면, KL-Divergence는 y_true(P)가 가지는 분포값과 y_pred(Q)가 가지는 분포값이 얼마나 다른 지를 확인하는 방법입니다. **KL-Divergence는 값이 낮을수록 두 분포가 유사하다고 해석**합니다. 정보이론에서 흔히 볼 수 있는 엔트로피(entropy) 또한, 값이 낮을수록 랜덤성이 낮다고 해석하는 것과 비슷합니다. KL-Divergence에 대한 자세한 설명은 대학원생이 쉽게 설명해보기님의 [KL-Divergence Loss 간단 설명](#https://hwiyong.tistory.com/408)를 참고하시길 바랍니다.

$$
\begin{aligned}
  D_{KL}(P||Q) &= H(P,Q) - H(P) \\
               &= (\sum_x p(x) \log q(x)) - (-\sum_x p(x) \log p(x)) \\
\end{aligned}
$$

*   $H(P, Q)$: $P$ 대신 $Q$를 사용할 때의 cross-entropy
*   $H(P)$: 원래의 $P$ 분포가 가지는 entropy 

따라서, 본 연구에서는 두 분포 soft assignments $q_{ij}$와 target distribution $p_{ij}$의 차이를 최소화하는 방향으로 학습한다는 것을 알 수 있습니다.

다음으로, target distibutions P를 구하는 것은 DEC의 성능에 있어서 중요한 요소로 작용합니다. $q_i$는 진짜 label이 아닌 unsupervised setting으로 계산된 확률이므로 $p_i$역시 softer probabilistic targets을 사용하는 것이 자연스럽다고 합니다.

특히 논문 저자들은 타겟 분포(target distribution)가 다음과 같은 특징을 갖고 있길 희망하였습니다.
1. 예측 강화
2. 높은 신뢰도(high confidence)로 할당된 data points에 더 강조
3. cluster 사이즈가 클수록 손실 함수에 주는 기여도가 커져서 전체 feature space를 왜곡시킬 수 있으므로 이를 방지하기 위하여 손실함수 값을 cluster 사이즈로 정규화합니다.

따라서 보조 타겟 분포(auxiliary target distribution)는 다음과 같이 정의됩니다.

$$
  p_{ij} = \frac{q_{ij}^2 / f_j}{\sum_{j'}q_{ij'}^2 / f_{j'}}
$$

$f_j = \sum_i q_{ij}$로, sample i가 cluster j에 속할 확률의 합을 나타냅니다. 
나눠주어 normalization 합니다.

#### OPIMIZATION

***



## Experiments

***

### Datasets

***

1개의 text dataset "REUTERS"와 2개의 image datasets "MNIST" 및 "STL-10"에 대하여 DEC의 성능을 평가하였습니다. 

![Dataset statistics](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/paper_review/2022-06-09-DEC/dataset_statistics.PNG?raw=true){:.aligncenter} 
<center><span style="color:gray; font-size:80%">본 논문에서 사용한 데이터셋 정보</span></center>

### Evaluation Metric

***

Unsupervised learning의 성능을 평가하고 다른 알고리즘과 비교하기 위하여 standard unsupervisd evaluation metric 및 procols을 사용하였습니다.

$$
  ACC = \displaystyle\max_m\frac{\sum_{i=1}^{n} 1\lbrace l_i = m(c_i)\rbrace}{n}  
$$


### Implementation








⇒ 즉, 보조 타겟 분포가 label이 되어 마치 supervised learning인 것처럼 학습하여 parameter를 최적화함

          
## Summary

본 논문은 feature space와 cluster memberships을 동시에 해결하는 방법을 제안하였습니다. 이를 위해 data space X에서 cluster에 최적화된 feature space Z로 parameterized non-linear mapping을 정의하였으며, clustering에 최적화된 mapping을 학습하기 위해 stochastic gradient descent(SGD)를 사용하였습니다. Deep Embedded Clustering(DEC)는 당시의 SOTA를 달성하였습니다.
   



## References

***

[1] Xie, Junyuan, Ross Girshick, and Ali Farhadi. "Unsupervised deep embedding for clustering analysis." International conference on machine learning. PMLR, 2016. [[Paper]](#http://proceedings.mlr.press/v48/xieb16.html)   
[2] 슈퍼짱짱, "[논문] DEC 리뷰: Unsupervised Deep Embedding for Clustering Analysis" [[Online]](#https://leedakyeong.tistory.com/entry/%EB%85%BC%EB%AC%B8Unsupervised-Deep-Embedding-for-Clustering-AnalysisDEC)    

<br>
<br>

***

<center>오류나 틀린 부분이 있을 경우 언제든지 댓글 혹은 메일로 지적해주시면 감사하겠습니다.</center>