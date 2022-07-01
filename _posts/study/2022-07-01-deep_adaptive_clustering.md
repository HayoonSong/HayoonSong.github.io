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
last_modified_at: '2022-07-01'
---

본 논문은 2017년 CVPR에 실렸으며 

- Table of Contents
{:toc .large-only}

## Overview

***

## Introduction

***

전통적으로 K-means 및 병합 군집(agglomerative clustering)과 같은 다양한 클러스터링 기법들이 연구되었습니다. 기존 방법은 이미지 데이터셋에서 식별하기 어려운 사전 정의된 distance metrics에 의존합니다. 최근에는 이미지의 표현(representation)을 학습하기 위해 autoencoder 및 auto-encoding Variational bayes와 같은 deep unsupervised feature learning 방법이 관심을 받고 있습니다. Deep unsupervised feature learning은 multi-stage pipeline을 채택하여, 먼저 unsupervised로 deep neural networks(DNN)을 사전학습하고 후처리로써 이미지 클러스터링을 위한 기존 방법을 사용합니다. 그러나 이러한 representation 기반 접근 방식은 multi-stage 패러다임의 번거로움과, 학습된 representation은 unsupervised feature learning 이후에 고정된다는 점에서 한계가 있습니다. 결과적으로 클러스터링 과정에서 representation은 더 이상 개선될 수 없습니다.

본 연구는 이미지 클러스터링을 위한 single-stage ConvNet 기반 방법인 Deep Adaptive Clustering을 제안합니다. 이 가정을 기반으로, image clustering task를 binary pairwise-classification model로 변환합니다.
 한 쌍의 이미지가 같은 클러스터에 속하는지 아닌지 판단하기 위한 binary pairwise-classification 문제로 여깁니다. 

### Contributions

* DAC 

## Deep Adaptive Clustering Model

***
먼저, pairwise 이미지의 관계를 이진법(binary)라고 가정합니다. 즉, 각 쌍의 이미지는 같은 클러스터에 속하거나 다른 클러스터에 속합니다. 이미지들 간의 유사도(similarities)는 알 수 없으므로, 본 연구는 



먼저 pairwise 이미지의 관계가 이진법이라고 가정합니다. 즉, 각 이미지 쌍은 동일한 클러스터 또는 다른 클러스터에 속합니다. 이 가정을 기반으로 이미지 클러스터링 작업을 이진 쌍별 분류 모델로 다시 변환합니다. 이미지 간의 유사도를 알 수 없기 때문에 유사도를 조사하여 모델을 학습시키기 위해 페어와이즈 이미지를 적응적으로 선택합니다. DAC의 흐름도는 그림 2에 나와 있습니다. 자세한 내용은 다음 하위 섹션에서 설명합니다.

### Binary Pairwise-Classification for Clustering

학습 데이터는 $$D = {(x_i,x_j,r_{ij})}_{i=1,j=1}^n$$로 나타내며, $$x_i, xj \in $$ 

DAC의 목적 함수(objective function)은 다음과 같이 정의됩니다.

$$
  \min{w} E(w) = \sum_{i,j} L(r_{ij},g(x_i,x_j;w))
$$

* $$L(r_{ij},g(x_i,x_j;w))$$: $$r_{ij}$$와 추정된 유사도 $$g(x_i,x_j;w)$$간의 loss
* w: 모델 파라미터
