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
published: false

last_modified_at: '2022-06-08'
---

Tensorflow를 사용하여 시계열 데이터를 증강하는 기법에 대해 알아보겠습니다.

![Overview](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/2022-06-01-augmentation/overview.png?raw=true){:.aligncenter}
<center><span style="color:gray; font-size:80%">본 포스팅에서 구현할 데이터 증강 기법</span></center>   
<br>

- Table of Contents
{:toc .large-only}

## Data augmentation

***

데이터 증강(data augmentation)은 데이터의 양을 늘리거나 다양한 데이터에 강건한 모델을 만들고자 사용됩니다.

예제로는 시계열 데이터인 EEG data를 사용하겠습니다.   

![Raw signal](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/2022-06-01-augmentation/raw.png?raw=true){:.aligncenter}
<center><span style="color:gray; font-size:80%">상: 전체 데이터 하: 1초 확대한 데이터</span></center>   
<br>

22 channels EEG 데이터는 4초간 측정되었으며, 250 Hz의 sampling frequency가 사용되었습니다.   
데이터 차원은 [22 x 1000]이지만 자세한 설명을 위하여 
4초 중에서 1초만 확대하고 하나의 채널(single-channel)만 시각화 하겠습니다.

### Amplitude scale

***

진폭 스케일(amplitude sclae)은 신호에 상수를 곱하여 **진폭의 크기를 조정**하는 기법입니다.

![Amplitude scale](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/2022-06-01-augmentation/amplitude_scale.png?raw=true){:.aligncenter}

~~~python
def amplitude_scale(signal, num_scale):
  signal = num_scale * signal
  return signal
~~~

### Time shift

***

시간 이동(time shift)은 말 그대로 **시간 축으로 이동**한다는 것입니다.   
Temporal roll이라고 불리기도 하며, 원래의 시간축에서 오른쪽 방향으로만 이동하는   
시간 지연(temporal delay)도 포함됩니다. 

![Time shift](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/2022-06-01-augmentation/time_shift.png?raw=true){:.aligncenter}  

~~~python
import tensorflow as tf

# 시간 단위가 아닌 sample 단위로 계산하였습니다.
# 예제 데이터는 sampling rate 250 Hz로 4초간 측정되었기에, SAMPLES = 1000 입니다.
# num_plces_to_shift는 t0와 동일하며, 어디 시점에서 신호를 굴릴 것인지
# 즉 신호가 이동되는 시작점을 나타냅니다.
def time_shift(signal, num_places_to_shift):
  assert abs(num_places_to_shift) <= signal.shape[-1]

  signal = tf.roll(signal, num_places_to_shift, axis=-1)
  return signal
~~~

[tf.roll](#https://www.tensorflow.org/api_docs/python/tf/roll)은 축(axis)에 따라 signal의 num_places_to_shift에서부터 신호를 이동시킵니다.   
num_places_to_shift가 음수일 경우 앞으로 양수일 경우 뒤로 이동하며, 지연된 신호를 원하신다면 양수를 넣으면 됩니다.

### DC shift

***

DC 이동(DC shift)는 신호에 상수를 더하여 **진폭(amplitude)을 이동**하는 방법입니다.

![DC shift](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/2022-06-01-augmentation/dc_shift.png?raw=true){:.aligncenter}  

~~~python
def dc_shift(signal, num_amplitude_to_shift):
  signal = num_amplitude_to_shift + signal
  return signal
~~~

### Temporal cutout

***

Temporal cutout은 시계열 신호의 특정 구간을 0으로 만들어 zero-masking이라고도 합니다.

![Temporal cutout](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/2022-06-01-augmentation/temporal_cutout.png?raw=true){:.aligncenter}  

~~~python
import numpy as np

def temporal_cutout(signal, t0, t):
  SAMPLES = signal.shape[-1]
  indices = np.arange(SAMPLES)
  indices[t0: t0+t] = -1
  mask = tf.one_hot(indices, depth=SAMPLES, dtype=signal.dtype)
  
  # 1차원 시계열 데이터는 행렬벡터 곱연산
  if tf.rank(signal) == 1:
    return tf.linalg.matvec(mask, signal)
  
  # 2차원 시계열 데이터는 행렬 곱연산
  return tf.linalg.matmul(signal, mask)
~~~

tf.one_hot은 one-hot 인코딩하는 tensorflow 함수로 자세한 설명은 [이전 포스팅](#https://hayoonsong.github.io/study/2022-02-11-tf/)을 참고해주시길 바랍니다.

1차원 시계열 데이터는 벡터 행렬곱을 통해서, 2차원 시계열 데이터는 행렬곱을 통해서 cutout할 특정 구간을 0으로 만들 수 있습니다.

### Gaussian noise

***

기존 데이터에 가우시안 잡음(Gaussian noise)를 추가하여 데이터를 변형시킬 수 있습니다.

![Gaussian noise](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/2022-06-01-augmentation/gaussian_noise.png?raw=true){:.aligncenter}

~~~python
def gaussian_noise(signal, sigma):
  # Adding Gaussian noise
  noise = tf.random.normal(shape=tf.shape(signal), stddev=sigma, dtype=signal.dtype)
  signal = tf.add(signal, noise)
  return signal
~~~

### Band-stop filter

***

Band-stop 필터는 다른 말로 notch filter 또는 band-reject filter라고 하며, **특정한 주파수 대역만을 차단**하는 역할을 합니다.

이전 포스팅에서 `scipy` 모듈을 활용하여 FFT 변환 과정을 살펴보고 Band-pass filter 및 Band-stop filter를 구현하였습니다.

그러나, scipy를 사용하여 tensorflow의 tensor를 필터링하고자 할 때 tf.Tensor가 numpy로 계산되어 다음과 같은 에러가 났습니다.

"NotImplementedError: Cannot convert a symbolic tf.Tensor (args_2:0) to a numpy array. 
This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported."

`.numpy()` 또는 `.eval()` 등 tf.Tensor를 numpy로 변경할 수 있는 몇 가지 방법이 있었지만 제 텐서는 변하지 않았습니다...

따라서 최대한 tensorflow의 내장 함수를 이용하여 band-stop filter를 구현하였습니다.
그러나 tensorflow에서 butterworth 함수를 발견하지 못하였기에 부드러운 필터링을 대신하여 이상적 대역저지 필터(Ideal Band-stop Filter)를 만들었습니다.
혹시 tensorflow 내장 함수를 사용하여 butterworth band-pass filter를 구현한 코드를 발견하신다면 댓글 또는 메일 부탁드립니다.

저는 tensorflow 내에서 band-stop filter가 반드시 필요하여 scipy와 최대한 비슷한 신호가 나오게 구현하도록 노력했지만,
scipy를 대체하지 못하기에 하단의 코드를 추천드리지는 않습니다...

![Band-stop filter all](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/2022-06-01-augmentation/bandstop_filter_all.png?raw=true){:.aligncenter}

Original signal, scipy 기반 band-stop filtering, 제가 구현한 tensorflow 기반 band-stop filtering 결과를 비교해보면, tensorflow 기반 band-stop filter가 scipy와 일치하지 않은 것을 확인하실 수 있습니다.

![Band-stop filter](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/2022-06-01-augmentation/bandstop_filter.png?raw=true){:.aligncenter}
<center><span style="color:gray; font-size:80%">상: scipy 기반 band-stop filtering 하: tensorflow로 구현한 band-stop filtering</span></center>   
<br>

![Band-stop filter FFT](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/2022-06-01-augmentation/bandstop_filter_fft.png?raw=true){:.aligncenter}
<center><span style="color:gray; font-size:80%">위에서부터 첫 번째: Original signal의 FFT 적용 결과 <br>
두 번째: scipy 기반 band-stop filter와 FFT 적용 결과 세 번째: tensorflow 기반 band-stop filter와 FFT 적용 결과</span></center>
<br>

또한, Original signal, scipy 기반 band-stop filtering, 제가 구현한 tensorflow 기반 band-stop filtering한 신호들을 FFT를 사용하여 주파수 신호로 변환해보면 scipy를 기반으로 한 두 번째 그림은 부드럽게 특정 대역(20 - 40 Hz)이 차단된 반면에, 제가 구현한 세 번째 그림은 갑자기 신호가 끊긴 듯한 느낌이 듭니다. 

~~~python
from scipy import fft

  def bandstop_filter(signal, sfreq, lowcut, highcut):
    SAMPLES = signal.shape[-1]
    signal = tf.cast(signal, dtype=tf.complex64)
    fft_signal = tf.signal.fft(signal) / SAMPLES
    freqs = fft.fftfreq(SAMPLES, d=1/sfreq)

    mask = np.arange(SAMPLES)
    bandstop_frequency = np.intersect1d(np.where(lowcut <= abs(freqs)),
                                        np.where(abs(freqs) <= highcut))
    mask[bandstop_frequency] = -1
    mask = tf.one_hot(mask, SAMPLES, dtype=tf.complex64)

    # 1차원 시계열 데이터는 행렬벡터 곱연산
    if tf.rank(signal) == 1:
      filtered = tf.linalg.matvec(mask, fft_signal)
    # 2차원 시계열 데이터는 행렬 곱연산
    else:
      filtered = tf.linalg.matmul(fft_signal, mask)

    filtered_signal = tf.signal.ifft(filtered) * SAMPLES
    filtered_signal = tf.cast(filtered_signal, dtype='float64')
    return filtered_signal
~~~

### Crop and upsample

***

마지막으로 Crop and upsample은 데이터를 특정 부분 자르고 업샘플링하여 타임스탬프의 빈도를 늘리는 방법입니다.

![Crop and upsample compairson](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/2022-06-01-augmentation/crop_upsample_compairson.png?raw=true){:.aligncenter}
<center><span style="color:gray; font-size:80%">상: Original signal 하: Crop and upsample을 적용한 transformed signal </span></center>
<br>

원래는 임의로 특정 부분을 정하지만, 시각화를 위하여 t0 = 0으로 설정하고 4초 데이터 중에서 뒤에 2초를 crop하고 앞에 2초를 upsampling 해보았습니다.
Original signal의 0~2초가 crop and upsampling을 통해 4초로 늘어난 것을 확인할 수 있습니다.

~~~python
def crop_and_upsample(signal, crop_samples):
  SAMPLES = signal.shape[-1]
  DELAY = int(0.1 * crop_samples)
  t0 = np.random.randint(0, SAMPLES)
  remain_samples = SAMPLES - crop_samples
  if t0 + remain_samples > SAMPLES:
    # 1차원 시계열 데이터
    if tf.rank(signal) == 1:
      signal = tf.tile(signal, multiples=[2])
    # 2차원 시계열 데이터
    else:
      signal = tf.tile(signal, multiples=[1, 2])

  indices = np.arange(t0, t0 + remain_samples + DELAY)
  cropped_signal = tf.gather(signal, indices, axis=-1)
  cropped_signal = tf.cast(cropped_signal, dtype=tf.float32)
  upsampled_signal = tfio.audio.resample(tf.transpose(cropped_signal),
                                         remain_samples+DELAY,
                                         SAMPLES+DELAY)
  upsampled_signal = tf.cast(tf.transpose(upsampled_signal), dtype=tf.float64)
  upsampled_signal = tf.gather(upsampled_signal, np.arange(DELAY, upsampled_signal.shape[-1]), axis=-1)
  return upsampled_signal
~~~

`tf.tile`은 tensor를 복사하여 붙여넣는 함수로 자세한 설명은 [이전 포스팅](#https://hayoonsong.github.io/study/2022-02-11-tf/)을 참고해주시길 바랍니다. Crop하는 시작점 즉 t0이 신호의 끝 쪽에 있어서 원하는 samples만큼 자르지 못할 때 신호를 복붙하여 늘려줍니다.

DELAY는 [tfio.audio.resample](#https://www.tensorflow.org/io/api_docs/python/tfio/audio/resample) 함수를 사용하여 resampling할 경우    
데이터의 앞부분을 제대로 resampling 하지 못하기에 추가하였습니다.
DELAY를 사용하지 않게 되면 신호는 다음과 같이 변환됩니다.

![Transformed](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/2022-06-01-augmentation/crop_upsample_delay/delay_transformed.png?raw=true){:.aligncenter}

0 ~ 0.25초 정도까지는 제대로 예측하지 못하는 것을 확인할 수 있습니다.

![Delay](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/2022-06-01-augmentation/crop_upsample_delay/delay.png?raw=true){:.aligncenter}
<center><span style="color:gray; font-size:80%"></span></center>
<br>

DELAY를 사용하지 않고 crop & resample한 신호를 original signal과 1초만 확대해서 비교해보면 신호를 제대로 복원하지 못한 것을 확연히 알 수 있습니다.

따라서 데이터의 앞부분을 제대로 복원하기 위해 자르고 싶은 샘플 개수의 0.1배 만큼을 더 남겨두었습니다.

`DELAY = int(0.1 * crop_samples)`

![Original](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/2022-06-01-augmentation/crop_upsample_delay/delay_original.png?raw=true){:.aligncenter}   
e.g.) TIME = 4, SAMPLING_RATE= 250 → SAMPLES = 1000     
      만약 자르고 싶은 부분이 2초라면 → crop_samples = 500, DELAY = 50    

`tfio.audio.resample`은 float64를 지원하지 않으므로 upsampling하기 전에 데이터 타입을 float32로 변경하였습니다.

Upsampling 단계에서는 신호의 앞부분을 제대로 복원하지 못하는 것을 감안하여 DELAY만큼 더 많이 upsamling 합니다.   
![Transformed](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/2022-06-01-augmentation/crop_upsample_delay/delay_transformed2.png?raw=true){:.aligncenter}

결과적으로 앞에 제대로 복원하지 못한 DEALY는 자르고 남은 데이터를 사용하였습니다.

![Crop and upsample](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/2022-06-01-augmentation/crop_upsample.png?raw=true){:.aligncenter}
<br>

이로써, Tensorflow를 사용하여 데이터를 증강하는 7가지의 기법에 대해 알아보았습니다.


## References

***

[1] Cheng, Joseph Y., et al. "Subject-aware contrastive learning for biosignals." arXiv preprint arXiv:2007.04871 (2020). [[Paper]](#https://arxiv.org/abs/2007.04871)   
[2] Mohsenvand, Mostafa Neo, Mohammad Rasool Izadi, and Pattie Maes. "Contrastive representation learning for electroencephalogram classification." Machine Learning for Health. PMLR, 2020. [[Paper]](#http://proceedings.mlr.press/v136/mohsenvand20a.html)   
[3] Han, Jinpei, Xiao Gu, and Benny Lo. "Semi-Supervised Contrastive Learning for Generalizable Motor Imagery EEG Classification." 2021 IEEE 17th International Conference on Wearable and Implantable Body Sensor Networks (BSN). IEEE, 2021. [[Paper]](#https://ieeexplore.ieee.org/abstract/document/9507038)

<br>
<br>

***

<center>오류나 틀린 부분이 있을 경우 언제든지 댓글 혹은 메일로 지적해주시면 감사하겠습니다.</center>
