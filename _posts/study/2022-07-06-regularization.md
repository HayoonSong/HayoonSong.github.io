---
layout: post
title: '[Deep learning] L1 and L2 Regularization'
description: >
  L1 및 L2 정규화의 특성
subtitle: Lasso and Ridge Regularization
date: '2022-07-06'
categories:
    - study
comments: true
published: true
last_modified_at: '2022-07-06'
---

정규화(regularization)은 알고리즘의 **일반화(generalization)을 개선**하기 위해 사용되는 기법입니다. 이번 포스팅에서는 L1 및 L2 정규화 다른 말로 Lasso 및 Ridge 정규화에 대해 알아보고자 합니다.

* this unordered seed list will be replaced by the toc
{:toc}

## Overview

***

|              | L1 norm<br>(Lasso)                                                                                                              | L2 norm<br>(Ridge)                                                                                                            |
|:------------:|---------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
|     수식     |                                                       $${\lVert w \rVert}_1$$                                                       |                                                $$\frac{1}{2} {\lVert w \rVert}^2$$                                                |
|     특성     | - 가중치의 값을 완전히 0으로 축소<br>- 중요한 특징을 선택하는 feature selection 효과<br>- Convex하여 global optimum에 수렴 가능 | - 가중치의 값을 0에 가까운 수로 축소<br>- 모델의 전반적인 복잡도를 감소시키는 효과<br>- Convex하여 global optimum에 수렴 가능 |
| 선택<br>기준 | 전반적으로 features가 비슷한 수준으로<br>성능에 영향을 미치는 경우                                                              | Features의 영향력 편차가 큰 경우                                                                                              |


## Regularization

***

정규화(regularization)는 weights에 페널티를 줌으로써 **predict function에 복잡도를 조정**하는 작업입니다. 학습 데이터에 과적합(overfitting)되는 것을 방지하고 모델의 강건함을 개선하기 위해 사용됩니다. 즉, Loss function에 regularization을 더하여 학습 데이터에 편중되어 학습하는 것을 방지하게 합니다. 따라서, Loss + Regularization은 **제약 조건이 있는 상태의 최적화** 문제입니다.

정규화는 일반화(genenralization)을 위해서 제약 조건을 추가하는 기법으로, **Loss 값이 감소하는 것을 기대하면 안됩니다.**

![Regularization](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/paper_review/2022-07-04-similarity/regularization.PNG?raw=true)   
{:.figure}

### L1 regularization

$$cost(W,b) = \frac{1}{m} \sum_i^m L(^{y}_i,y_i) + \lambda \frac{1}{2} \lvert w \rvert$$

$$
Cost function &= Loss + L1 Weight Penalty \\
              &= \sum_{i=1}^M {(y_i - \sum_{j=1}^N x_{ij}w_j)}^2 + \color{red} \lambda \sum_{j=1}^N \lvert w_j \rvert
$$

### L2 regularization

$$cost(W,b) = \frac{1}{m} \sum_i^m L(^{y}_i,y_i) + \lambda \frac{1}{2} {\lvert w \rvert}^2$$

$$
Cost function &= Loss + L2 Weight Penalty \\
              &= \sum_{i=1}^M {(y_i - \sum_{j=1}^N x_{ij}w_j)}^2 + \color{red} \lambda \sum_{j=1}^N w_j^2
$$

$$\lambda$$는 정규화 비중을 얼마나 줄 것인지 정하는 계수입니다. 0에 가까울수록 정규화의 효과는 사라집니다. K-fold cross validation을 통해 적절한 $$\lambda$$ 값을 찾을 수 있습니다.
 

![Lasso and Ridge regression](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/paper_review/2022-07-04-similarity/manhattan/l1_l2.PNG?raw=true)     
A 3-D plot of Iris dataset(Source: [Wikipedia](https://en.wikipedia.org/wiki/Lasso_(statistics)#/media/File:L1_and_L2_balls.svg))).
{:.figure}

* Sparsity(희소성) & Feature selection(변수 선택)
  + Sparse vectors란 벡터 내 대부분의 값이 0인 것(e.g., one-hot vectors)
  + L1을 적용할 경우
    - Vectors가 sparse vectors로 되는 경향이 있음
    - 즉, 작은 weights의 값은 0이 되므로 weights의 수를 줄이고 small set을 만들 수 있음
    - 다시 말해 weights가 너무 많을 때 좀 더 중요한 weights만 선택하고, 나머지는 weights는 0으로 되는 효과를 얻을 수 있음
    - 이러한 특징으로 feature selection이 가능함
  + L2를 적용할 경우
    - 모든 가중치가 균등하게 작아짐
    - 일반적으로 학습시 더 좋은 성능을 보임
  + 예를 들어, 두 vectors a와 b가 있음   
    a = (0.25, 0.25, 0.25, 0.25)
    b = (-0.5, 0.5, 0, 0)
    - 두 벡터의 L1 norm
      $${\lVert a \rVert}_1 = abs(0.25) + abs(0.25) + abs(0.25) + abs(0.25) = 1$$
      $${\lVert b \rVert}_1 = abs(-0.5) + abs(0.5) + abs(0.0) + abs(0.0) = 1$$
    - 두 벡터의 L2 norm
      $${\lVert a \rVert}_2 = \sqrt{0.25^2 + 0.25^2 + 0.25^2 + 0.25^2} = 0.5$$
      $${\lVert b \rVert}_2 = \sqrt{(-0.5)^2 + (0.5}^2 + 0^2 + 0^2} = 0.7071$$
    - L2는 각 vectors 값에 대해 unique한 값이 출력되는 반면, L1은 경우에 따라 특정 feature(vector의 요소)없이도 같은 값을 낼 수 있음

## References

***

[1] Towards Data Science, 17 types of similarity and dissimilarity measures used in data science. [[Online]](https://towardsdatascience.com/17-types-of-similarity-and-dissimilarity-measures-used-in-data-science-3eb914d2681)

<br>

***

<center>오류나 틀린 부분이 있을 경우 언제든지 댓글 혹은 메일로 지적해주시면 감사하겠습니다.</center>