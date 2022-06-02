---
layout: post
title: '[EEG] Signal data augmentation in Tensorflow'
subtitle: signal augmentation
date: '2022-05-31'
categories:
    - study
tags:
    - signal, eeg, audio
comments: true
pusblished: false

last_modified_at: '2022-06-01'
---

Tensorflow를 사용하여 시계열 데이터를 증강하는 기법에 대해 알아보겠습니다.

- Table of Contents
{:toc .large-only}

## Data augmentation
데이터 증강(data augmentation)은 데이터의 양을 늘리거나 다양한 데이터에 강건한 모델로 학습하기 위하여 사용됩니다.

예제로는 시계열 데이터인 EEG data를 사용하겠습니다.   

![Raw signal](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/2022-06-01-augmentation/raw_signal.jpg?raw=true){:.aligncenter}
<center><span style="color:gray; font-size:70%">좌: 전체 데이터 우: 1초 확대한 데이터</span></center>
</br>

22 channels EEG 데이터는 4초간 측정되었으며, 250 Hz의 sampling rate가 사용되었습니다.   
차원의 형태는 [22 x 1000]이지만 자세한 설명을 위하여 4초 중에서 1초만 확대하고 하나의 채널(single-channel)만 시각화 하겠습니다.

### Amplitude scale

진폭 스케일(amplitude sclae)은 신호에 상수를 곱하여 진폭의 크기를 조정하는 기법입니다.
![Amplitude scale](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/2022-06-01-augmentation/amplitude_scale.png?raw=true){:.aligncenter}

~~~js
def amplitude_scale(signal, num_scale):
  signal = num_scale * signal
  return signal
~~~

### Time shift (Temporal delay, Temporal roll)

시간 이동(time shift)은 말 그대로 시간 축으로 이동한다는 것입니다.   
Temporal roll이라고 불리기도 하며, 원래의 시간축에서 오른쪽 방향으로만 이동하는 시간 지연(temporal delay)도 포함됩니다.   
![Time shift](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/2022-06-01-augmentation/time_shift.png?raw=true){: width="50%" height="50%"}{:.aligncenter}  

~~~js
import tensorflow as tf

// 시간 단위가 아닌 sample 단위로 계산하였습니다.
// 예제 데이터는 sampling rate 250 Hz로 4초간 측정되었기에, SAMPLES = 1000 입니다.
// num_plces_to_shift는 t0과 동일하며, 어느 시점(samples)에서 신호를 굴릴 것인지 즉 신호가 이동되는 시작점을 나타냅니다.
def time_shift(signal, num_places_to_shift):
  assert abs(num_places_to_shift) <= SAMPLES
  
  signal = tf.roll(signal, num_places_to_shift, axis=-1)
  return signal
~~~

[tf.roll](#https://www.tensorflow.org/api_docs/python/tf/roll)은 축(axis)에 따라 signal의 num_places_to_shift에서부터 신호를 이동시킵니다.
num_places_to_shift가 음수일 경우 앞으로 양수일 경우 뒤로 이동하며, 지연된 신호를 원하신다면 양수를 넣으면 됩니다.

### DC shift
DC 이동(DC shift)는 신호에 상수를 더하여 진폭(amplitude)을 이동하는 방법입니다.
![DC shift](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/2022-06-01-augmentation/dc_shift.png?raw=true){: width="50%" height="50%"}{:.aligncenter}  

~~~js
def dc_shift(signal, num_amplitude_to_shift):
  signal = num_amplitude_to_shift + signal
  return signal
~~~

### Temporal cutout
Temporal cutout은 시계열 신호의 특정 구간을 0으로 만들며 zero-masking이라고도 합니다.
![Temporal cutout](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/2022-06-01-augmentation/temporal_cutout.png?raw=true){: width="50%" height="50%"}{:.aligncenter}  

~~~js
import numpy as np

def temporal_cutout(signal, t0, t):
  indices = np.arange(SAMPLES)
  indices[t0: t0+t] = -1
  mask = tf.one_hot(indices, depth=SAMPLES, dtype='float64')
  
  // 1차원 시계열 데이터일 경우
  if tf.rank(signal) == 1:
    return tf.linalg.matvec(mask, signal)
  return tf.linalg.matmul(signal, mask)

[tf.one_hot](#https://www.tensorflow.org/api_docs/python/tf/one_hot)은 one-hot 인코딩하는 함수입니다.
기본적으로는 아래와 같이 사용됩니다.
~~~js
tf.one_hot(indices=[0, 1, 2], depth=3)

// output: [3 x 3]
[[1., 0., 0.],
 [0., 1., 0.],
 [0., 0., 1.]]
~~~

tf.one_hot을 통해 단위행렬(identity matrix)에서 cutout할 구간의 인덱스를 -1로 만들어줍니다.

~~~js
np.arange(5) # output: [0, 1, 2, 3, 4]
tf.one_hot(indices=[0, -1, -1, 3, 4], depth=5)

// output: [5 x 5]
[[1., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0.],
 [0., 0., 0., 1., 0.],
 [0., 0., 0., 0., 1.]]
~~~

1차원 시계열 데이터는 벡터 행렬곱을 통해서, 2차원 시계열 데이터는 행렬곱을 통해서
cutout할 특정 구간을 0으로 변환시킬 수 있습니다.

### Gaussian noise (Noise, Noise addition, addictive Gaussian noise)
![Gaussian noise](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/2022-06-01-augmentation/gaussian_noise.png?raw=true){: width="50%" height="50%"}{:.aligncenter}

### Band-stop filter
![Band-stop filter](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/2022-06-01-augmentation/bandstop_filter.png?raw=true){: width="50%" height="50%"}{:.aligncenter}

### Crop and upsample
![Crop and upsample](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/2022-06-01-augmentation/crop_upsample.png?raw=true){: width="50%" height="50%"}{:.aligncenter}


## References

[1] Cheng, Joseph Y., et al. "Subject-aware contrastive learning for biosignals." arXiv preprint arXiv:2007.04871 (2020). [[Paper]](#https://arxiv.org/abs/2007.04871)   
[2] Mohsenvand, Mostafa Neo, Mohammad Rasool Izadi, and Pattie Maes. "Contrastive representation learning for electroencephalogram classification." Machine Learning for Health. PMLR, 2020. [[Paper]](#http://proceedings.mlr.press/v136/mohsenvand20a.html)   
[3] Han, Jinpei, Xiao Gu, and Benny Lo. "Semi-Supervised Contrastive Learning for Generalizable Motor Imagery EEG Classification." 2021 IEEE 17th International Conference on Wearable and Implantable Body Sensor Networks (BSN). IEEE, 2021. [[Paper]](#https://ieeexplore.ieee.org/abstract/document/9507038)
