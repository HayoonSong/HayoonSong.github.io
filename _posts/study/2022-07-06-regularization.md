---
layout: post
title: '[Deep learning] L1 and L2 Regularization'
description: >
  L1 및 L2 정규화의 특성
subtitle: Lasso and Ridge Regularization
date: '2022-07-05'
categories:
    - study
tags:
    - deep-learning
comments: true
published: true
last_modified_at: '2022-07-07'
---

정규화(regularization)는 알고리즘의 **일반화(generalization)을 개선**하기 위해 사용되는 기법입니다. 이번 포스팅에서는 L1 및 L2 정규화 다시 말해 Lasso 및 Ridge 정규화에 대해 알아보고자 합니다.

* this unordered seed list will be replaced by the toc
{:toc}

## Overview

***

|              | L1 norm (Lasso)<br>$${\lVert w \rVert}_1 = \lvert w \rvert$$                                                                    | L2 norm (Ridge)<br>$$\frac{1}{2} {\lVert w \rVert}^2 = \frac{1}{2}w^2$$                                                       |
|:------------:|---------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
|     특성     | - 가중치의 값을 완전히 **0으로 축소**하는 경향<br>- 중요한 특징을 선택하는 **feature selection 효과**<br>- Convex하여 global optimum에 수렴 가능 | - 가중치의 값을 **0에 가까운 수로 축소**하는 경향<br>- 모델의 **전반적인 복잡도를 감소**시키는 효과<br>- Convex하여 global optimum에 수렴 가능 |
| 선택<br>기준 | Features의 영향력 편차가 큰 경우                                                                                              | 전반적으로 features가 비슷한 수준으로<br>성능에 영향을 미치는 경우                                                              |


## Regularization

***

![Regularization](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/deep_learning/2022-07-06-regularization/regularization.jpg?raw=true)   
{:.figure}

정규화(regularization)는 weights에 페널티를 줌으로써 **predict function에 복잡도를 조정**하는 작업입니다. 학습 데이터에 과적합(overfitting)되는 것을 방지하고 모델의 강건함을 개선하기 위해 사용됩니다. 다시 말해 Loss function에 regularization을 더하여 학습 데이터에 편중되어 학습하는 것을 방지하게 합니다. 따라서, Loss + Regularization은 **제약 조건이 있는 상태의 최적화** 문제입니다.

정규화는 일반화(genenralization)를 위하여 제약 조건을 추가하는 기법으로, **Loss 값이 감소하는 것을 기대하면 안됩니다.**

### L1 regularization

$$
  \begin{align}
  Cost &= \text{Loss + L1 Weight Penalty} \\
       &= \sum_{i=1}^M (y_i - \sum_{j=1}^N x_{ij}w_j)^2 + \color{red} \lambda \sum_{j=1}^N \lvert w_j \rvert
  \end{align}
$$

$$\lambda$$는 정규화 비중을 얼마나 줄 것인지 정하는 계수입니다. **0에 가까울수록 정규화의 효과는 사라집니다.** K-fold cross validation을 통해 적절한 $$\lambda$$ 값을 찾을 수 있습니다.

L1 regularization의 경우 Cost 함수를 가중치로 미분하게 되면 regularization term의 $$w_j$$들은 모두 1까지 미분되어 결국 $$\lambda$$라는 상수항만 남게 됩니다. 가중치 업데이트는 원래 가중치에서 Cost 함수를 가중치 $$w_j$$로 미분한 값과 학습률을 곱한 값이 빠지면서 이루어집니다. 결국 **고정된 상수값 $$\lambda$$가 모든 항에 대해 공평하게 빠지면서, 작은 가중치들은 0이 되며 중요한 가중치만 남아서 feautre 수가 줄어들고 과적합을 방지**할 수 있습니다.

L1 regularizatoion을 사용하는 선형 회귀 모델을 Lasso model이라고 합니다.

### L2 regularization

$$
  \begin{align}
  Cost &= \text{Loss + L2 Weight Penalty} \\
       &= \sum_{i=1}^M (y_i - \sum_{j=1}^N x_{ij}w_j)^2 + \color{red} \frac{1}{2} \lambda \sum_{j=1}^N w_j^2
  \end{align}
$$

$$\lambda$$ 앞에 있는 $$\frac{1}{2}$$은 실험적으로 알아내야 하는 값입니다. $$\frac{1}{2}$$: Cost를 미분하면 $$w_j^2$$에 있는 2가 내려오면서 $$\frac{1}{2}$$가 제거되고 regularization parameter인 $$\lambda$$만 남을 수 있게 됩니다. $$\frac{1}{2}$$을 하지 않는 경우도 있으며, 그 때는 regularization term으로 $$\lambda \sum_{j=1}^N w_j^2$$만 사용합니다.

L2 regularization의 경우 Cost 함수를 가중치로 미분하게 되면 regulariztion term의 $$w_j^2$$는 $$w_j$$로 미분되어 $$\frac{1}{2} \lambda w_j$$라는 일차항들이 남게 됩니다. 따라서 **각 가중치의 크기에 대해 비례하게 값이 빠지게 되며**, 가중치들이 완전히 0이 되지 않습니다. 가중치 업데이트는 Loss 값에 현재 가중치 크기의 일정 비율을추가로 반영하여, 현 시점에서의 가중치를 일정비율로 깎아 내리면서 이루어집니다. 결론적으로 L2는 크기가 **큰 가중치에 대한 규제는 강하게, 작은 가중치에 대한 규제는 약하게** 주게 됩니다. 이러한 성질로 L2 regularization을 기중치 감쇠(weight decay)라고도 합니다.

L2 regularization을 사용하는 선형 회귀 모델을 Ridge model이라고 합니다.

### Sparsity & Feature selection

* L1: 가중치의 값을 완전히 **0으로 축소**하는 경향 $$\to$$ feature selection 가능
* L2: 가중치의 값을 **0에 가까운 수로 축소**하는 경향

![L1 vs L2](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/deep_learning/2022-07-06-regularization/l1_l2.jpg?raw=true)   
{:.figure}

파라미터가 w1, w2로 2개일 때 L1 또는 L2 Regularization의 분포와 Acutal Error Gradient가 **한점에서 접하는 지점**이 **최적화 하는 weights의 값**이 됩니다. L1 regularization을 좌표평면에 나타내면 마름모 형태의 분포와 같고, 그림의 예제에서 $$w_1 = 0, w_2 > 0$$일 때 Loss의 최솟값을 가졌습니다. 즉 L1 regularization은 sparse vectors를 생성하는 경향이 있습니다. Sparse vector란 벡터 내 대부분의 값이 0인 벡터를 말합니다. 반면에 L2 regularization을 좌표평면에 나타내면 원의 형태와 같고, $$w_1 >0, w_2 > 0$$인 한 점에서 최적화 되었습니다. 이처럼 L2 regularization은 가중치를 0으로 만들기 보다는 0에 가까운 수로 축소합니다.

따라서, L1 regularization은 L2 regularization 보다 sparse vector(e.g., $$w_1 = 0, w_2 >0$$)를 만들 확률이 높습니다.

![sparsity](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/deep_learning/2022-07-06-regularization/sparsity.jpg?raw=true)   
{:.figure}

오른쪽 그림에서 알 수 있듯이 L2 regularization의 경우 원과 직선의 교점은 1개입니다. 따라서 Sparse vector가 생성되는 점 A에서 최적화 될 경우는 현저히 적습니다. 그러나 왼쪽 그림 L1 regularization의 경우 꼭지점과 만날 수 있는 직선의 수는 많으므로, 그만큼 sparse vector가 될 확률이 높아집니다.

L1 regularization은 sparse vectors를 생성하는 경향으로 인해 머신러닝에서 feature selection을 가능하게 합니다. 작은 weights의 값을 0으로 만들기에 weights가 너무 많을 때 좀 더 중요한 weights만 선택학 나머지 weights는 0으로 되는 효과를 얻을 수 있습니다.

L2 regularization은 sparse vectors를 만들지 못합니다. 왜냐하면 weights가 0에 가가울수록 미분값도 0에 아주 가까워므로 weight 업데이이트가 잘 되지 않기 때문입니다. 대신에 **모든 가중치를 균등하게 감소**시킵니다. 일반적으로 학습 시 더 좋은 결과를 만듭니다. L1은 절대값의 형태를 가지므로 미분이 불가능합니다. 이러한 이유로 실제 딥러닝에서는 L2 regularization를 주로 사용합니다.

### Convexity

* L1, L2: Convex하여 global optimum에 수렴 가능

일반적으로 사용하는 Loss 함수는 미분 가능한 형태로 역전파(backpropagation)가 가능하며 convex합니다. L1 및 L2 regularization도 loss와 동일하게 covex합니다. L1은 V 형태를 띄고 L2는 U 형태를 띄기 때문입니다. 최적화 이론에서 convex + convex = convex가 성립하므로 Loss(convex) + Regularization(convex) = Loss(convex)가 성립하게 됩니다. 그러나, U 형태의 L2와 달리 **L1은 V 형태이므로 미분 불가능**하기에 gradient-based learning에서는 주의할 필요가 있습니다.

## Summary

***

머신러닝은 직접 중요한 feature를 input으로 사용하므로 feature의 영향력에 따라 L1 또는 L2 regularization을 선택합니다. 반면에, 딥러닝은 task(e.g., regression, classification)에 있어 중요한 feature를 자동적으로 골라내야 하므로 주로 L2 regularization이 사용됩니다. 

## References

***

[1] Jinsol Kim, L1,L2 Regularization. [[Online]](https://gaussian37.github.io/dl-concept-regularization)   
[2] 별보는 두더지, L1, L2 Norm & L1, L2 loss & L1, L2 규제. [[Online]](https://mole-starseeker.tistory.com/34)   
[3] Seongkyun Han's blog, L1 & L2 loss/regularization. [[Online]](https://seongkyun.github.io/study/2019/04/18/l1_l2)


<br>

***

<center>오류나 틀린 부분이 있을 경우 언제든지 댓글 혹은 메일로 지적해주시면 감사하겠습니다.</center>