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

last_modified_at: '2022-06-10'
---

Clustering 기반의 self-supervised learning을 활용한 Deep Embedded Clustering(DEC)를 알아보고자 합니다. 설명에 앞서 슈퍼짱짱님의 블로그를 참고하였음을 밝힙니다.

- Table of Contents
{:toc .large-only}


## Introduction

***

## Related work

***

## Deep embeddded clustering

***

기존의 clustering은 data space X에서 다이렉트로 군집화하였다면, 본 논문에서 제안하는 Deep embedded clustering은 data space X에 있는 n points들은 non-linear mappling 함수 $f_θ$를 통해 latent space Z로 변환시키고(※ 이 때, "curse of dimensionality"에 따라 Z의 차원은 X보다 작아야합니다), latent space Z에서 clustering을 하게 합니다.

Deep embedded clustering (DEC)는 두 단계로 이루어져 있습니다.    
1. Parameter initialization with a deep autoencoder   
2. Parameter optimization (i.e., clustering)   
    * 보조 타겟 분포(auxiliary target distribution)의 계산 및 Kullback-Leibler (KL) divergence의 최소화를 반복함으로써 최적화시킵니다.

### Clustering with KL divergence

***

KL divergence 기반 clustering은 다음의 두 단계를 반복하여 이루어집니다.
Step 1. X → Z로 mapping된 embedded points와 cluster centroids 사이의 soft assignment 계산합니다.   
Step 2. 보조 타겟 분포(auxiliary target distribution)를 사용하여 높은 신뢰도(high confidence)로 non-linear mapping ($f_θ$)을 업데이트하고 cluster centroids를 재정의합니다.

#### SOFT ASSIGNMENT

***

embedded points $z_i$와 cluster centroids $M_j$ 사이의 유사도를 계산하기 위해 t 분포(Student's t-distribution)를 사용하였습니다.

![Soft Assignment](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/paper_review/2022-06-09-DEC/soft_assignment.PNG?raw=true){:.aligncenter} 
<br>

수식에 나와있듯이 embedded points $z_i$와 cluster centroids $M_j$ 사이의 유사도를 계산하기 위해 유클리드 거리 (Euclidean Distance)를 사용하여 두 점 사이의 거리를 구합니다. alpha 값은 t 분포의 degree of freedom을 나타내며, $q_ij$는 분모 부분의 normalize 과정을 통해 확률값의 형태로 표현됩니다. 즉, 샘플 i과 cluster j에 속할 확률을 나타냅니다. Clustering은 unsupervised setting으로 alpha를 validation set에 cross-validate하지 못하기에 모든 실험에서 alpha를 1로 설정했다고 합니다.

#### KL DIVERGENCE MINIMIZATION

***

본 저자들은 보조 타켓 분포(auxiliary target distribution)을 사용하여 높은 신뢰도(high confidence)로 군집을 조정하는 것을 제안하였습니다. 특히 soft assignmnet를 target distribution과 매칭하여 학습하였습니다. 따라서 soft assignment $q_i$와 auxiliary distribution $p_i$ 간의 KL divergence loss를 목적함수로 정의 하였습니다.

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

### Parameter initialization

***

Deep embedded clustering(DEC) 및 cluster centroids를 초기화(initialization) 한 방법에 대해 소개드리겠습니다.

* DEC로는 stacked autoencoder(SAE)를 사용하였습니다. SAE로 학습된 unsupervised representation  

## Experiments

***

### Datasets

***

1개의 text dataset과 2개의 image datasets에 대하여 DEC의 성능을 평가하였습니다. 




⇒ 즉, 보조 타겟 분포가 label이 되어 마치 supervised learning인 것처럼 학습하여 parameter를 최적화함

          

   



## References

***

[1] Xie, Junyuan, Ross Girshick, and Ali Farhadi. "Unsupervised deep embedding for clustering analysis." International conference on machine learning. PMLR, 2016. [[Paper]](#http://proceedings.mlr.press/v48/xieb16.html)   
[2] 슈퍼짱짱, "[논문] DEC 리뷰: Unsupervised Deep Embedding for Clustering Analysis" [[Online]](#https://leedakyeong.tistory.com/entry/%EB%85%BC%EB%AC%B8Unsupervised-Deep-Embedding-for-Clustering-AnalysisDEC)    

<br>
<br>

***

<center>오류나 틀린 부분이 있을 경우 언제든지 댓글 혹은 메일로 지적해주시면 감사하겠습니다.</center>