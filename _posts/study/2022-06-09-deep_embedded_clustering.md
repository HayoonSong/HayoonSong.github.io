---
layout: post
title: '[Paper Review] Unsupervised Deep Embedding for Clustering Analysis'
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

본 논문은 2016년 PMLR에 실렸으며 clustering 기반의 self-supervised learning을 활용한 Deep Embedded Clustering(DEC)을 제안하였습니다. 설명에 앞서 슈퍼짱짱님의 블로그를 참고하였음을 밝힙니다.

- Table of Contents
{:toc .large-only}


## Introduction

***

Clustering은 데이터 분석 및 시각화(visualization)에서 핵심적인 기법이며 unsupervised machine learning으로 연구되고 있습니다.

??본 논문에서는 현재의 soft cluster assignment에서 나온 보조 타겟 분포 (auxiliary target distribution)을 사용하여 clusters를 재정의하는 방법을 제안하였습니다.?? 이를 통해 feature representation과 더불어 clustering을 점진적으로 개선시켰습니다. 

또한 DEC는 이미지 및 텍스트 데이터셋에서 accuracy와 running time 모두 최고 성능(state-of-the-art, SOTA)을 달성하였다고 합니다.

## Contributions

***

* Deep embedding과 clustering의 공동 최적화
* Soft assignment를 통한 clusters 재정의
* 정확도 및 속도에서 SOTA clustering 달성


## Deep embeddded clustering

***

기존의 clustering은 data space X에서 clustering 하였다면, 본 논문에서 제안하는 Deep embedded clustering은 data space X에 있는 data를 latent space Z로 변환시키고 latent space Z에서 clustering을 하게 합니다("curse of dimensionality"에 따라 Z의 차원은 X보다 작아야합니다). 이 때, 모델은 **feature representations(X → Z mapping)과 clustering을 동시에 학습**합니다.

Deep embedded clustering (DEC)는 두 단계로 이루어져 있습니다.    
1. Parameter initialization with a deep autoencoder   
2. Parameter optimization (i.e., clustering)   
    * 보조 타겟 분포(auxiliary target distribution)의 계산 및 Kullback-Leibler (KL) divergence의 최소화를 반복하여 최적화시킵니다.

### Parameter initialization

***

DNN parameters $θ$와 cluster centroids ${μ_j}$를 초기화하는(initialize) 방법을 알아보겠습니다.

DEC network의 $θ$를 초기화하기 위하여 **Stacked denoising autoencoder**가 활용되었습니다.
Stacked autoecoder는 여러개의 히든 레이어를 가지는 오토인코더이며, 레이어를 추가할수록 오토인코더가 더 복잡한 코딩(부호화)을 학습할 수 있게 됩니다. 여기에서 각 레이어는 denoising autoencoder로 구성되어 noise를 추가한 입력으로부터 noise가 없는 원본 입력을 재구성하도록 학습되었습니다. Stacked autoencoder 및 denoising autoencoder를 포함하여 autoencoder에 대한 자세한 설명은 Excelsior-JH님의 [오토인코더 (AutoEncoder)](#https://excelsior-cjh.tistory.com/187)를 참고하시길 바랍니다.

<br>

![Network structure](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/paper_review/2022-06-09-DEC/network_structure.PNG?raw=true){:.aligncenter}<center><span style="color:gray; font-size:80%">Network structure</span></center>   
<br>

이렇게 학습된 Stacked denoising autoencoder에서 encoder 부분을 DEC network의 초기 파라미터로 사용하였습니다.

다음으로 $k$개의 cluster centroids {$μ_j$}$_{j=1}^k$의 초기값으로는 앞서 초기화된 네트워크를 통해 나온 embedded data를 k-means clustering하여 사용하였습니다.

### Parameter optimization

***

Non-linear mapping $f_θ$과 cluster centroids {$μ_j$}$_{j=1}^k$의 초기값을 추정하였으므로, 비지도 알고리즘을 사용하여 clustering을 개선하는 방법을 살펴보겠습니다.

#### Clustering with KL divergence

***

KL divergence 기반 clustering은 다음의 두 단계를 반복하여 이루어집니다.   
**Step 1.** X → Z로 mapping된 embedded points $z_i$와 cluster centroids $M_j$ 사이의 **soft assignment(샘플 $i$가  cluster $j$에 속할 확률)를 계산**합니다.   
**Step 2.** 보조 타겟 분포(auxiliary target distribution)를 사용하여 높은 신뢰도(high confidence)로 non-linear mapping ($f_θ$)을 업데이트하고 cluster centroids를 재정의합니다. ⇒ 즉, **보조 타겟 분포를 label로 사용**하여 supervised learning를 통해 파라미터를 최적화하며, 이 절차를 수렴 기준에 충족될 때까지 반복합니다.

#### SOFT ASSIGNMENT

***

Embedded points $z_i$와 cluster centroids $μ_j$ 사이의 유사도를 계산하기 위해 유클리드 거리 (Euclidean Distance)를 사용하여 두 점 사이의 거리를 구합니다. 

![Soft Assignment](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/paper_review/2022-06-09-DEC/soft_assignment.PNG?raw=true){:.aligncenter} 
<br>

Alpha 값은 t 분포(Student's t-distribution)의 degree of freedom을 나타내며, $q_{ij}$는 분모 부분의 normalize 과정을 통해 확률값의 형태로 표현됩니다. 즉, $p_{ij}$는 샘플 i가 cluster j에 속할 확률을 나타냅니다. Clustering은 비지도 알고리즘으로써 alpha를 validation set에 cross-validate하지 못하므로 모든 실험에서 alpha를 1로 설정하였습니다.

#### KL DIVERGENCE MINIMIZATION

***

다음으로 저자들은 보조 타켓 분포(auxiliary target distribution)을 사용하여 높은 신뢰도(high confidence)로 clusters를 재정의하였습니다. 특히 soft assignment를 target distribution과 매칭하여 학습하였습니다. 따라서 soft assignment $q_i$와 auxiliary distribution $p_i$ 간의 KL divergence loss를 목적함수로 정의하였습니다.

![KL Divergence](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/paper_review/2022-06-09-DEC/kl_divergence.PNG?raw=true){:.aligncenter} 
<br>

Target distibutions P를 구하는 것은 DEC의 성능에 있어서 중요한 요소로 작용합니다. 단순하게 생각해보면 cluster의 centroids 주변에 일정 threshold를 넘은 data points는 delta 분포로 여기고, threshold를 넘지 못한 나머지 데이터는 무시하는 방법이 있습니다. 

그러나 $q_i$는 진짜 label이 아닌 unsupervised setting으로 계산된 확률이므로 $p_i$역시 softer probabilistic targets을 사용하는 것이 자연스럽다고 합니다.

특히 논문 저자들은 타겟 분포(target distribution)가 다음과 같은 특징을 갖고 있길 희망하였습니다.
1. 예측 강화
2. 높은 신뢰도(high confidence)로 할당된 data points에 더 강조
3. cluster 사이즈가 클수록 손실 함수에 주는 기여도가 커져서 전체 feature space를 왜곡시킬 수 있으므로 이를 방지하기 위하여 손실함수 값을 cluster 사이즈로 정규화합니다.

따라서 보조 타겟 분포(auxiliary target distribution)는 다음과 같이 계산됩니다.

![Auxiliary target distribution](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/paper_review/2022-06-09-DEC/auxiliary_target_distribution.PNG?raw=true){:.aligncenter} 
<br>

$f_i$는 cluster j에 있는 data의 개수이며, 각 cluster j 마다 $f_j$로 나눠주어 normalization 합니다.

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

![Unsupervised clustering accuracy](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/paper_review/2022-06-09-DEC/acc.PNG?raw=true){:.aligncenter} 

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