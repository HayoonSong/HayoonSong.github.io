---
layout: post
title: '[Paper Review] Subject-Aware Contrastive Learning for Biosignals'
subtitle: Contrastive learing
date: '2022-03-15'
categories:
    - study
tags:
    - contrastive learning
comments: true
published: true

last_modified_at: '2022-03-15'
---

본문은 Constrastive learning 기반 Self-supervised leanring을 활용하였습니다.

## Abstract

***

<Biosignals의 한계>
Datasets for biosignals, such as electroencephalogram (EEG) and electrocardiogram(ECG), 
often have **noisy labels** and have **limited number of subjects (<100)**.

* subject-invariance loss:
improves representation quality for these tasks
* subject-specific loss:
increases performance when fine-tuning with supervised labels

## 1. Introduction

***

<기여도>
* Apply self-supervised learning to biosignals
* Develop data augmentation techniques for biosignals:
In this work, we develop **domain-inspired augmentation** techniques. 
For example, the power in certain EEG frequency bands has been shown to be highly correlated with different brain activities. 
Thus, we use **frequency-based perturbations** to augment the signal.
* Integrate subject awareness into the self-supervised learning framework


## 3. Methods

***

### 3-1. Model and Contrastive learning


## 4. Experiments and Results

***

### 4-1. EEG: PhysioNet Motor Imagery Dataset

* Subjects: 106 명 (3명 데이터 제외)
* Class: 4 개(closing the right fist, closing the left fist, closing both fists, and moving both feet)
* Time: 4 sec
* Channels: 64 개
* Sampling rate: 160 Hz
* Preprocessing:
  - Re-referening: channel average
  - Normalization: mean and standard deviation

Experimental Setup.

Encoder G:
* 90 명의 데이터를 사용하여 self-supervised learing 
* Input은 4 초의 시간 중에서 랜덤으로 2초를 선택하여 320 samples을 사용함
* 256-dimensional embedding vector
* Batch size: 400, steps: 270000
* 모델 끝에 logistic-regression lienar classifier를 추가하여 90 명의 데이터에 대한 정확도를 평가함
* 데이터는 cue 다음 0.25초 이후의 신호를 사용함
* Intersubject testing을 위해 나머지 16명의 데이터에 평가함

Model F:

Downstream taks + classifier:
* 2-class: rigth fist, left fist
* 4-class: right fist, left fist, both fists, both feet
* Setting:
    - Learning rate: 1e-3
    - Batch size: 256
    - Epochs: 2000

Impact of Transformation.


Impact of Subject-Aware Training.

* No augmentation
* Base SSL
* Subject-specific
* Subject-invariant
* Random encoder: Randomly-initialized encoder was used as a baseline for comparison


