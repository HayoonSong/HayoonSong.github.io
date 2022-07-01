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
    - representation-learning
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

전통적으로 K-means 및 병합 군집(agglomerative clustering)과 같은 다양한 클러스터링 기법들이 연구되었습니다. 기존 방법은 이미지 데이터셋에서 식별하기 어려운 사전 정의된 distance metrics에 의존합니다. 최근에는 이미지의 표현(representation)을 학습하기 위해 autoencoder 및 auto-encoding Variational bayes와 같은 deep unsupervised feature learning 방법이 관심을 받고 있습니다. Deep unsupervised feature learning은 multi-stage pipeline을 채택하여, 먼저 unsupervised 방법으로 deep neural networks(DNN)을 사전학습하고 후처리로써 이미지 클러스터링을 위한 기존 방법을 사용합니다.