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

정규화(regularization)는 알고리즘의 **일반화(generalization)을 개선**하기 위해 사용되는 기법입니다. 이번 포스팅에서는 L1 및 L2 정규화 다시 말해 Lasso 및 Ridge 정규화에 대해 알아보고자 합니다.

* this unordered seed list will be replaced by the toc
{:toc}

## Overview

***

|              | L1 norm (Lasso)<br>$${\lVert w \rVert}_1 = \lvert w \rvert$$                                                                    | L2 norm (Ridge)<br>$$\frac{1}{2} {\lVert w \rVert}^2 = \frac{1}{2}w^2$$                                                       |
|:------------:|---------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
|     특성     | - 가중치의 값을 완전히 0으로 축소하는 경향<br>- 중요한 특징을 선택하는 feature selection 효과<br>- Convex하여 global optimum에 수렴 가능 | - 가중치의 값을 0에 가까운 수로 축소하는 경향<br>- 모델의 전반적인 복잡도를 감소시키는 효과<br>- Convex하여 global optimum에 수렴 가능 |
| 선택<br>기준 | Features의 영향력 편차가 큰 경우                                                                                              |                                                                                           | 전반적으로 features가 비슷한 수준으로<br>성능에 영향을 미치는 경우                                                              |


## Regularization

***

![Regularization](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/deep_learning/2022-07-06-regularization/regularization.jpg?raw=true)   
{:.figure}

정규화(regularization)는 weights에 페널티를 줌으로써 **predict function에 복잡도를 조정**하는 작업입니다. 학습 데이터에 과적합(overfitting)되는 것을 방지하고 모델의 강건함을 개선하기 위해 사용됩니다. 즉, Loss function에 regularization을 더하여 학습 데이터에 편중되어 학습하는 것을 방지하게 합니다. 따라서, Loss + Regularization은 **제약 조건이 있는 상태의 최적화** 문제입니다.

정규화는 일반화(genenralization)을 위해서 제약 조건을 추가하는 기법으로, **Loss 값이 감소하는 것을 기대하면 안됩니다.**

![L1 vs L2](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/deep_learning/2022-07-06-regularization/l1_l2.jpg?raw=true)   
{:.figure}


파라미터가 w1, w2로 2개일 때, 왼쪽 그림은 L1 regularization을 오른쪽 그림은 L2 regularization을 나타냅니다.

### L1 regularization

$$
  \begin{align}
  Cost &= \text{Loss + L1 Weight Penalty} \\
       &= \sum_{i=1}^M {(y_i - \sum_{j=1}^N x_{ij}w_j)}^2 + \color{red} \lambda \sum_{j=1}^N \lvert w_j \rvert
  \end{align}
$$

* 가중치의 값을 완전히 0으로 축소
  + 


### L2 regularization

$$
  \begin{align}
  Cost &= \text{Loss + L2 Weight Penalty} \\
       &= \sum_{i=1}^M {(y_i - \sum_{j=1}^N x_{ij}w_j)}^2 + \color{red} \frac{1}{2} \lambda \sum_{j=1}^N w_j^2
  \end{align}
$$



$$\lambda$$는 정규화 비중을 얼마나 줄 것인지 정하는 계수입니다. 0에 가까울수록 정규화의 효과는 사라집니다. K-fold cross validation을 통해 적절한 $$\lambda$$ 값을 찾을 수 있습니다.

![Regularization](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/deep_learning/2022-07-06-regularization/regularization.jpg?raw=true)   
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
      $${\lVert b \rVert}_2 = \sqrt{(-0.5)^2 + (0.5)^2 + 0^2 + 0^2} = 0.7071$$
    - L2는 각 vectors 값에 대해 unique한 값이 출력되는 반면, L1은 경우에 따라 특정 feature(vector의 요소)없이도 같은 값을 낼 수 있음

## References

***

[1] Jinsol Kim, L1,L2 Regularization. [[Online]](https://gaussian37.github.io/dl-concept-regularization)
[2] Seongkyun Han's blog, L1 & L2 loss/regularization. [[Online]](https://seongkyun.github.io/study/2019/04/18/l1_l2)

<br>

***

<center>오류나 틀린 부분이 있을 경우 언제든지 댓글 혹은 메일로 지적해주시면 감사하겠습니다.</center>