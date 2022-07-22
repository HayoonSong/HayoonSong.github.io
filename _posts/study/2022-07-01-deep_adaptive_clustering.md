---
layout: post
title: '[Paper Review] Deep Adaptive Image Clustering'
description: >
  거꾸로 읽는 self-supervised learning의 두 번째 논문
subtitle: Deep Adaptive Clustering
date: '2022-07-01'
categories:
    - study
tags:
    - self-supervised-learning
comments: true
published: false
last_modified_at: '2022-07-06'
---

본 논문은 2017년 ICCV에 개제되었으며

* this unordered seed list will be replaced by the toc
{:toc}

## Overview

***

Image clustering 문제를 Binary pairwise-classification 프레임워크로 전환하여 접근 

![The flowchart of DAC](https://cdn.jsdelivr.net/gh/HayoonSong/Images-for-Github-Pages/study/paper_review/2022-07-01-DAC/DAC_flowchart.PNG?raw=true)   
The flowchart of Deep Adaptive Clustering
{:.figure}


* Framework:
  - Step 1. Generate label features: Input인 unlabeled images를 **ConvNet을 통해 label features로 변환**(즉, 이미지를 representation vectors로 변환)
  - Step 2. Cacluate cosine similarities: Cosine distance로 label features 간의 **유사성 추정**
  - Step 3. Select labeled samples: Cosine similarity를 기반으로 유사성이 모호한 데이터들은 제외하고, 값이 1인 것은 similar한 이미지 쌍으로, 값이 0인 것은 dissimilar한 이미지 쌍으로 여기며 training samples 선택
  - Step 4. Train the ConvNet: 선정된 training samples로 ConvNet 학습

* Model:
* Loss:
* Task:
* Contribution:

## Introduction

***

클러스터링 기법으로는 전통적으로 K-means 및 병합 군집(agglomerative clustering)과 같은 다양한 기법들이 연구되었습니다. 기존에는 사전 정의된 distance metrics(e.g, Euclidean Metric)를 활용하여 클러스터링하였지만, 이미지 데이터셋에서 distance 기반 척도로는 이미지간의 유사성을 구별하기 어렵다는 한계가 있습니다. (예를 들어, 이미지 픽셀 벡터를 기반으로 pixel MSE가 낮으면 같은 얼굴이고 높으면 다른 얼굴일까요? 아니죠!)   

최근에는 Deep unsupervised feature learning 기법(e.g., autoencoder, auto-encoding Variational bayes)이 이미지의 표현 학습(representation learning)에서 관심을 받고 있습니다. Deep unsupervised feature learning은 multi-stage pipeline으로 형성되어 먼저 unsupervised로 deep neural networks(DNN)을 사전학습하고 후처리로써 기존의 방법으로 이미지를 클러스터링 합니다. 그러나 이러한 representation 기반 접근 방식은 multi-stage 패러다임의 번거로움과, 학습된 representation은 unsupervised feature learning 이후에 고정된다는 점에서 한계가 있습니다. 결과적으로 클러스터링 과정에서 representation은 더 이상 개선될 수 없습니다.

본 연구는 이미지 클러스터링을 위한 single-stage ConvNet 기반 방법인 Deep Adaptive Clustering(DAC)을 제안하였습니다. Image clustering task를 **한 쌍의 이미지가 같은 클러스터에 속하는지 아닌지 판별하는 binary pairwise-classification 문제**로 여깁니다. 구체적으로 이미지는 deep ConvNet으로 생성된 label features로 표현되고, 유사성은 label features간의 cosine distance로 측정됩니다. 또한, DAC에 제약조건을 추가하였으며 이를 통해 학습된 label feautres는 one-hot vectors가 되는 경향이 있습니다. 본 연구에서는 cosine distance로 추정된 유사성을 사용하였으며 실제 유사성은 알 수 없습니다. 따라서 모델의 최적화를 위해 alternating iterative 방법인 Adaptive Learning 알고리즘을 개발하였습니다. 

During each iteration, pairwise images with the estimated similarities are first selected based on the fixed ConvNet. Subsequently, DAC employs the selected labeled samples to train the ConvNet in a supervised way. The algorithm converges when all the samples are included for training and the objective function of the binary pairwise-classification problem can not be improved further. Finally, images are clustered by locating the largest response of label features. The visual results of DAC on the MNIST test set are illustrated in Figure 1. To sum up, the main contributions of this work are:

### Contributions

* DAC 

## Deep Adaptive Clustering Model

***

먼저, pairwise 이미지 간의 관계를 이진법(binary)으로 가정합니다. 즉, 각 쌍의 이미지는 같은 클러스터에 속하거나 다른 클러스터에 속합니다. 이 가정을 기반으로 image clustering task를 binary pairwise-classifiation 모델로 변환합니다. 

![The flowchart of DAC](https://cdn.jsdelivr.net/gh/HayoonSong/Images-for-Github-Pages/study/paper_review/2022-07-01-DAC/DAC_flowchart.PNG?raw=true)   
The flowchart of DAC
{:.figure}

### Binary Pairwise-Classification for Clustering

***

학습 데이터를 $$D = {(x_i,x_j,r_{ij})}_{i=1,j=1}^n$$로 표현할 때 $$x_i, x_j \in X$$는 unlabeled 이미지이며, $$r_{ij} \in Y$$은 unknown binary variable입니다. $$r_{ij} = 1$$이면 $$x_i$$와 $$x_j$$가 같은 클러스터에, $$r_{ij} = 0$$이면 $$x_i$$와 $$x_j$$가 다른 클러스터에 속한다는 것을 나타냅니다.

DAC의 목적 함수(objective function)은 다음과 같이 정의됩니다.

$$
  \min_w E(w) = \sum_{i,j} L(r_{ij},g(x_i,x_j;w)), \tag{1}
$$

* $$g(x_i,x_j;w)$$: 추정된 유사도(estimated similarity), 즉 모델이 학습하여 추론한 $$x_i$$와 $$x_j$$의 유사도
* $$L(r_{ij},g(x_i,x_j;w))$$: Label $$r_{ij}$$와 추정된 유사도 $$g(x_i,x_j;w)$$간의 loss
* $$w$$: 모델 파라미터

따라서 Loss는 다음과 같이 정의됩니다.

$$
  L(r_{ij},g(x_i,x_j;w)) = -r_{ij}log(g(x_i,x_j;w)) - (1-r_{ij})log(1-g(x_i,x_j;w)). \tag{2}
$$

Binary Cross Entropy Loss와 동일한 구조입니다. 그러나 상단의 수식에는 두 가지 문제가 있다고 합니다.   
1. 추정된 유사도 $$g(x_i, x_j ;w)$$만을 가지고 $$x_i$$와 $$x_j$$의 클러스터를 알 수 없음
2. 이미지 클러스터링 과정에서 y 즉 $$r_{ij}$$를 알 수 없음   
Section 3.2 및 3.3에서 두 가지 문제의 해결방법에 대한 설명을 이어가겠습니다.

### Label Fetures under Clustering Constraint

***

이미지 쌍의 유사도를 추정하기 위해 label features $$L = {\lbrace l_i \in \Bbb{R}^k \rbrace}_{i=1}^n$$가 도입되었습니다. $$l_i$$는 이미지 $$x_i$$의 k-dimensional label feature를 나타냅니다.
**유사도 $$g(x_i, x_j ;w)$$는 두 label features 간의 cosine distance**로 정의되었습니다. 또한, 이미지 클러스터링에 유용한 feature representation을 학습하기 위해 **label features에 clustering constraint를 추가**하였습니다.

$$
  \forall \, i, \, \lVert l_i \rVert_2 = 1, and l_{ih} \geq 0, \, h = 1,\dots,k, \tag{3}
$$

* $$l_i$$: 이미지 $$x_i$$의 k-dimensional label feature
* $$\lVert \cdot \rVert_2$$: L2-norm
* $$l_{ih}$$: Label feature $$l_i$$의 h번째 요소

즉 i 개의 모든 이미지 데이터가 있을 때, 각 **label features의 L2 norm이 1**이 되도록 제약을 둔 것입니다.   

벡터의 L2 norm을 1로 만들려면 L2 normalization을 하면 됩니다. 따라서, 저는 이 제약을 L2 normalizaion 했다는 것으로 해석하였습니다. L2 norm을 하게 되면 $$\lVert l_i \rVert_2 = 1$$ 이 되면서 본 연구에서 도입한 clustering constraint를 따르게 됩니다. L2 normalization에 대한  설명은 이전 포스팅 [L1 and L2 Norm / Normalization / Regularization](https://hayoonsong.github.io/study/2022-07-05-L1_L2/)을 참고하시길 바랍니다. 머신러닝의 전처리에서 정규화(normalization)를 하는 것과 같이 L2 normalization은 데이터의 값을 scaling 하여 원활한 학습에 도움을 줍니다. 또한, 데이터의 값을 0 ~ 1 사이의 값으로 만들어줌으로써 두 label features 간의 내적도 1 이하의 값이 되고 그래야 식 (2)의 $$log(1-g(x_i,x_j;w))$$ 값도 0 ~ 1 사이의 값이 될 수 있습니다.   
{:.faded}

i 개의 모든 이미지 데이터에서 각 label features의 L2 norm은 1이므로($$\forall i, \, \lVert l_i \rVert_2 = 1$$), cosine similarity $$g(x_i, x_j ;w)$$는 다음과 같이 계산할 수 있습니다.

$$g(x_i,x_j;w) = f(x_i;w) \cdot f(x_j;w) = l_i \cdot l_j, \tag{4}$$

* $$f_w$$: 입력 이미지를 label features로 매핑해주는 mapping function
* 연산자 $$\cdot$$ \,: 내적

즉, 두 이미지간의 유사도 $$g(x_i, x_j ;w)$$는 label features 간의 내적으로 정의할 수 있습니다. 

원래 cosine similarity는 두 vector의 내적을 각 L2 norm의 곱으로 나누어야 합니다.
$$cosine \, similarity := cos(\theta) = \frac{A \cdot B}{\lVert A \rVert \lVert B \rVert}$$      
그러나 본 연구에서는 clutering constraint 즉 L2 norm이 1이 되도록 만들어주었기에 label features 간의 내적이 곧 cosine similarity가 됩니다.   
{:.faded}

따라서 DAC 모델은 다음과 같이 재구성할 수 있습니다.

$$
  \begin{align}
  \min\limits_w E(w) = \sum_{i,j} L(r_{ij},l_i \cdot l_j), \tag{5} \\
  s.t.   \forall \, i, \, \lVert l_i \rVert_2 = 1, and l_{ih} \geq 0, \, h = 1,\dots,k.
  \end{align}
$$


상단의 식에서 clustering constraint는 데이터 클러스터링의 흥미로운 특징을 제공합니다. $$\Bbb{E}^k$$를 k-차원 유클리드 공간(Euclidean space)의 표준 기반이라고 하면 다음 정리를 따릅니다.

THEOREM 1. If the optimal value of upper equation is attained, for $$\; \forall i, \; j, \; l_i \in \Bbb{E}^k, \; l_i \not = l_j$$



### Labeled Training Samples Selection

**

$$
  r_{ij} := \begin{cases}
         1, \enspace \text{if } l_i \cdot l_j \geq u(\lambda), \\
         0, \enspace \text{if } l_i \cdot l_j \lt l(\lambda), \quad i, \; j=1, \dots, n, \\
         None, \enspace otherwise,
  \end{cases}
$$
