---
layout: post
title: '[Paper Review] Adversarial Discriminative Domain Adaptation'
subtitle: domain adaptation
date: '2022-01-03'
categories:
    - study
tags:
    - domain_adaptation
comments: true
pusblished: true

last_modified_at: '2022-01-03'
---

본 논문은 unsupervised domain adaptation에 GAN loss function을 도입하였습니다.


## Overview

***

## Abstract

***

### Adversarial learning method

1.  다양한 도메인에서 complex samples을 생성할 수 있습니다.
2.  도메인 이동(doamin shift) 또는 데이터셋 편향(dataset bias)에도 불구하고 인식 성능을 향상시킬 수 있습니다.
3.  Training 및 test domain의 분포 간 차이를 감소시켜 일반화 성능(generalization performance)을 향상시킬 수 있습니다.

### Adversarial Discriminative Domain Adaptation (ADDA)
*   Discriminative modeling
*   Untied weights sharing
*   GAN loss

## Introduction

***

대규모 데이터셋에서 학습된 Deep convolution networks는 다양한 과제(task) 및 시각 도메인(visual domains)에서 일반적으로 유용한 표현(representation)을 학습할 수 있습니다.

Problem
: 대규모 데이터셋으로 학습한 인식 모델은 데이터셋 편향 또는 도메인 이동으로 인해 새로운 데이터셋 또는 과제에 잘 일반화되지 않는다는 문제가 있습니다.   

Solution
: 
*   Fine-tune: 네트워크를 task-specific dataset에서 추가로 미세조정하는 방법으로, 충분한 labeled data를 얻기에 어렵고 비용이 크다는 단점이 있습니다.
*   Doamin adaption: 두 도메인을 공통된 특징 공간(feature space)으로 mapping하는 deep neural transformations을 학습합니다. 일반적으로 최대 평균 불일치(maximum mean discrepancy) 또는 상관 거리(correlation distances)와 같은 일부 측정(measure)을 최소화하기 위해 표현을 최적화함으로써 달성됩니다. 대안은 소스 표현(source representation)에서 대상 도메인(target domain)을 재구성하는 것입니다.
    -   Adversarial adaptation: 도메인 판별자(domain discriminator)에 대한 적대적 목적함수(adversarial objective)를 통해 대략적인 도메인 불일치 거리를 최소화하는 방법입니다. GAN과 비슷한 방식으로써 domain adpation에서는 네트워크가 training과 test domain 예제의 분포를 구별할 수 없도록 하기 위해 사용되었습니다.

Proposed method
Adversarial Discriminative Domain Adaptation (ADDA)
1.  Source domain의 label을 활용하여 discriminative representation을 학습합니다.
2.  Domain-adversarial loss를 통해 학습된 비대칭 매핑(asymmetric ampping)을 활용하여 target data를 동일한 공간에 매핑하는 별도의 인코딩(a separate encoding)을 학습합니다.

Contribution
1.   MNIST, USPS 및 SVHN 숫자 데이텃에서 SOTA visual adaptation을 달성하였습니다.
2.  RGB 컬러 이미지에서 깊이 관찰로 객체 분류기를 전송하여 인스턴스 제약 없이 훨씬 더 어려운 cross-modality shifts 사이의 격차를 메울 수 있는 가능성을 테스트하였습니다.
3. 표준 Office adaptation 데이터셋을 평가하고 ADDA가 competing mehtod 특히 가장 어려운 domain shift에서 강력한 개선을 달성하였습니다.

## Related work

***

Labeled source datasets에서 labeled datat가 적거나 존재하지 않은 target domain으로 DNN representation을 전이하는 데 중점을 두었습니다. Unlabeled target domain의 경우, 주요 전략은 source와 target feature 분포 간의 차이를 최소화하여 특징 학습(feature learning)을 유도하는 것이었습니다.
*   이를 위해 두 도메인 평균 간의 차이의 norm을 계산하는 Maximum Mean Discrepancy (MMD) loss를 사용하였습니다.
    -   Deep domain confusion (DDC) 방법은 source에 대한 regular classificaiton loss에 더하여 MMD를 사용하여 discriminative 및 domain invarint 표현을 학습합니다.
    -   Deep Adaptation Network (DAN)은 kernel Hilbert space에 포함된 layers에 MMD를 적용하여 두 분포의 고차 통계를 효과적으로 일치시켰습니다.
    -   Correlation Alignment (CORAL) 방법은 두 분포의 평균과 공분산을 일치시키기 위해 제안되었습니다.


*   도메인 이동을 최소화하기 위하여 adversarial loss를 선택하여 도메안을 구별할 수 없는 동시에 source label을 구별하는 표현을 학습합니다.
*   다른 연구에서는 입력의 이진 도메인 레이블을 분류하는 domain classificer (a single fully connected layer)를 추가하는 것을 제안하고 이진 레이블에 대한 균일한 분포에 가능한 한 근접하도록 예측을 장려하기 위해 domain confustion loss를 설계하였습니다.  
*   Gradient reversal algorithm (ReverseGrad)도 binary 분류 문제로서 도메인 불변성(domain invariance)을 잊니 분류 문제로 취급하지만 기울기를 반전시켜 domain 분류기의 loss를 직접 최대화 합니다.
*   Deep reconstruction-classifiation networks (DRCCN)

## Generalized adversarial adaptation

***

### Source and target mappings

### Adversarial losses

## Adversarial discriminative domain adaptation

***

## Experiments

***
Three digits datasets

### MNIST, UPSP, and SVHN digits datasets
SVHN과 MNIST 사이의 domain adaptation을 다른 연구와 비교하기 위해 전체 훈련 세트를 사용합니다. 모든 실험은 대상 도메인의 레이블을 사용하지 않는 설정에서 수행되며 MNIST->, USPS, USPS-> MNIST 및 SVHN->MNIST의 세 가지 방향의 domain adaptation을 고려합니다.

LeNet 구조를 활용
Adversarial discriminator는 3개의 fully connected layers로 구성되어 있으며, 2 layers는 500개의 hidden units으로 이루어져 있습니다.

### Modality adaptation
