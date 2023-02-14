---
layout: post
title: '[Paper Rivew] Batch Normalization Embeddings for Deep Domain Generalization'
subtitle: Domain generalization
date: '2022-03-31'
categories:
    - study
tags:
    - batch normalization, domain generalization, domain-specific normalization statistics
comments: true
published: true

last_modified_at: '2022-03-31'
---

본 논문은 batch normalization을 통한 domain generalization을 해결하고자 하였습니다.

## Method

The core idea of our method is to exploit domain-specific batch normalization statistics 
to map known and unknown domains in a shared latent space, 
where domain membership of samples can be measured according to their distance
from the domain embeddings of the known domains.

> 본 연구의 핵심 아이디어는 domain-specific 배치 정규화 통계를 활용하여
laten space에서 known domains과 unkonwn domain을 매핑하는 것입니다. 
where domain membership of samples can be measured according to their distance
from the domain embeddings of the known domains.


the goal of our method is to accurately estimate it as a mixture (i.e. linear combination) of the learned source distributions
> 방법의 목표는 Unkown domain의 sample이 주어졌을 때 y(output space)를 **학습된 source distributions의 mixture로 정확하게 추정**하는 것입니다. 

As opposed to the domain adaptation setting, we assume that target samples are not available at training time, 
and that each of them might belong to a different unseen domain.
> Domain adapation과 달리, 연구진들은 target samples을 학습하는 동안에는 사용하지 않았고 samples 각각은 서로 다른 domain에 속할 수 있다고 가정하였습니다.