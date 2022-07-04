---
layout: post
title: '[Deep learning] Similarity measure'
description: >
  머신러닝에서 사용되는 데이터 거리 측정 방법
subtitle: Similarity measure
date: '2022-07-04'
categories:
    - study
comments: true
published: true
last_modified_at: '2022-07-04'
---

머신러닝에서 데이터간의 거리를 구하는 방법에 대해서 알아보고자 합니다. 설명에 앞서 Mahmoud Harmouch님의 블로그를 번역하였음을 밝힙니다.

* this unordered seed list will be replaced by the toc
{:toc}

## Overview

***

![Overview](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/paper_review/2022-07-04-similarity/overview.PNG?raw=true)   
Various ML metrics. 출처: https://towardsdatascience.com/17-types-of-similarity-and-dissimilarity-measures-used-in-data-science-3eb914d2681
{:.figure}

## Introduction

***

데이터 사이언스에서 유사도 측정(similarity measure)은 개별 데이터 간의 관련성(relation)을 측정하는 방법입니다. 반면에 비유사도 측정(dissimilarity measure)은 개별 데이터들이 얼마나 구별되(distinct)를 나타냅니다. 유사도 측정이라는 용어는 머신러닝에서 다방면으로 사용됩니다.

* Clustering: 비슷한 데이터끼리 하나의 클러스터로 묶음
* Classification: 특징의 유사성(features's similarity)에 따라 데이터 분류(e.g., KNN)
* Anomaly detection: 다른 데이터들과 비교하여 유사하지 않은 이상값 탐지

일반적으로 유사도 측정은 숫자 값(numerical value)으로 표현됩니다. 데이터들 간의 관련성이 높아질 경우 값도 커집니다. 가끔 변환되어 0과 1사이의 숫자로 표현되기도 합니다. 0은 낮은 유사도, 즉 유사하지 않음을 뜻하며 1은 높은 유사도, 즉 매우 유사함을 의미합니다. 예를 들어, 하나의 input feature만을 갖고 있는 데이터 포인트 A, B, C가 있습니다. Input feature가 한 개이므로 각 데이터는 한 축에 하나의 값을 가질 수 있습니다. 그 축을 x축이라고 할 때, A(0.5), B(1), C(30)의 두 점을 살펴보겠습니다. 쉽게 알 수 있듯이, A와 B는 C와 비교할 때 서로 가깝습니다. 따라서, A와 B의 유사도는 A와 C 또는 B와 C의 유사도보다 높습니다. 다시 말해 **데이터 간의 거리가 작을수록, 유사도는 커집니다.**

### Metric

***

주어진 거리(distance)는 다음 네 가지 조건을 충족하는 경우에만 지표(metric)가 됩니다.

1. **Non-negativity**: d(p, q) ≥ 0, for any two distinct obeservations p and q.
2. **Symmetry**: d(p, q) = d(q, p) for all p and q.
3. **Triangle Inequality**: d(p, q) ≤ d(p, r) + d(r, q) for all p, q, r.
4. d(p, q) = 0 if only if p = q.

거리 측정은 주어진 데이터들 간의 비유사도를 측정하는 k-nearest neigbor(K-NN) 알고리즘과 같은 분류(classification)의 기본적인 원칙입니다. 또한, 어떤 거리 지표(distance metric)를 사용하느냐에 따라 분류기의 성능은 크게 달라집니다. 따라서, 객체 간의 거리를 구하는 방식은 분류기 알고리즘 성능에 중요한 역할을 합니다.

## Distance function

***

거리를 측정하기 위해 사용되는 기술은 작업 중인 상황에 따라 상이합니다. 예를 들어, 일부 영역에서는 유클리드 거리(Euclidian distance)가 거리 계산에 적절하고 유용할 수 있지만, 다른 영역에서는 코사인 거리(Cosine distance)와 같은 정교한 방식이 필요할 수 있습니다. 따라서, 데이터 포인트 간의 거리를 계산하는 다양한 방법을 소개하고자 합니다.

### L2 norm, Euclidean distance

***

Numeric attributes 또는 features와 같이 수치로 표현된 자료에서 가장 흔히 사용되는 거리 함수는 유클리드 거리(Euclidean distance)이며 수식은 다음과 같이 정의됩니다.

$$
  d(P,Q) &= \lVert P - Q \rVert_0 
         &= \sqrt{\sum_{i=1}^n (p_i - q_i)^2}
         &= \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2 + \dots + (p_n - q_n)^2}
$$
where:
$$P = (p_1,p_2,\dots,p_n),$$ and $$Q = (q_1,q_2,\dots,q_n)$$

아시다시피 유클리드 거리 측정법은 **대칭성(symmetry)**, **미분가능성(differentiability)**, **볼록성(convexity)**, **구형성(sphericity)** 등 잘 알려진 속성을 나타냅니다.

2차원 공간에서는 다음과 같이 표현할 수 있습니다.

$$
  d(P,Q) &= \lVert P - Q \rVert_0 &= \sqrt{\sum_{i=1}^n (p_i - q_i)^2}
                                  &= \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2}
$$
where:
$$P = (p_1,p_2),$$ and $$Q = (q_1,q_2)$$

해당 식은 직각삼각형의 빗변의 길이를 구하는 공식과 동일합니다.

또한, 유클리드 거리는 앞서 말씀드린 4가지 기준을 충족하기에 지표(metric)라고 할 수 있습니다.

![Euclidean distance](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/paper_review/2022-07-04-similarity/euclidean_distance.PNG?raw=true)   
The Euclidean distance satisfies all the conditions for being a metric. 출처: https://towardsdatascience.com/17-types-of-similarity-and-dissimilarity-measures-used-in-data-science-3eb914d2681
{:.figure}

또한, 해당 공식을 통해 계산된 거리는 **두 점 사이의 가장 짧은 거리**를 나타냅니다. 다시 말해, A 지점에서 B 지점으로 가는 최단 경로입니다.

![Shortest path](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/paper_review/2022-07-04-similarity/euclidean_distance_shortest_path.PNG?raw=true)   
The Euclidean distance is the shortest path(Excluding the case of a wormhole in a quantum world). 출처: https://towardsdatascience.com/17-types-of-similarity-and-dissimilarity-measures-used-in-data-science-3eb914d2681
{:.figure}

따라서, 유클리드 거리는 장애물이 없는 상태에서 두 점 사이의 거리를 계산해야 할 때 유용합니다. 가장 유명한 분류 알고리즘 중 하나인 KNN 알고리즘은 데이터를 분류하기 위해 유클리드 거리를 사용하여 이점을 얻을 수 있습니다. KNN의 분류 과정을 설명하고자 Scipy 패키지에 있는 iris dataset을 예시로 사용하겠습니다.

![Iris dataset](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/paper_review/2022-07-04-similarity/iris_dataset.PNG?raw=true)   
Iris dataset for two types of flowers in two features’ space. 출처: https://towardsdatascience.com/17-types-of-similarity-and-dissimilarity-measures-used-in-data-science-3eb914d2681
{:.figure}

Iris dataset은 붓꽃의 3가지 종(Iris-Setosa, Iris-Versicolor, Iris-Virginica)에 대해 4가지의 features(꽃받침 길이, 꽃받침 너비, 꽃잎 길이, 꽃잎 너비)가 있는 데이터입니다. 따라서 각 데이터 포인트를 표현할 수 있는 4차원 공간이 있습니다. 원활한 설명을 위하여 2가지의 features 꽃잎 길이(petal length)와 꽃잎 너비(petal width)만 사용하고, label도 2가지 Iris-Setosa, Iris-Versicolor만 사용하겠습니다. 이런 식으로 x축과 y축에 각각 petal length와 petal width를 나타내는 2차원 공간의 데이터 포인트를 시각화 할 수 있습니다.

![Iris training data](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/paper_review/2022-07-04-similarity/iris_training_data.PNG?raw=true)   
Training dataset. 출처: https://towardsdatascience.com/17-types-of-similarity-and-dissimilarity-measures-used-in-data-science-3eb914d2681
{:.figure}

그러나, 직선이 아닌 곡선을 따르는 경로에서는 유클리드 거리가 유용하지 않을 수 있습니다.  

