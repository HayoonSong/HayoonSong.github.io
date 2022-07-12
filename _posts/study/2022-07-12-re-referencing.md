---
layout: post
title: '[EEG] Re-referencing the EEG signal in MNE-Python'
description: >
  Re-referencing 설명 및 방법
subtitle: Re-referencing
date: '2021-07-12'
categories:
    - study
tags:
    - eeg
comments: true
published: true
last_modified_at: '2021-07-12'
---

MNE에서 Re-referencing하는 방법을 알아보고자 합니다.

* this unordered seed list will be replaced by the toc
{:toc}

## EEG re-referencing   

***

MNE뿐만 아니라 EEGLAB을 포함하여 EEG 신호를 다룰 때 reference electrode을 바꾸는 re-referencing을 쉽게 만날 수 있습니다. Acitve electrode, reference electrode, ground electorde 간의 관계에 대한 설명은 이전 포스팅 [뇌파의 이해](https://hayoonsong.github.io/study/2021-01-05-understanding_eeg/)를 참고해주시길 바랍니다.

Re-referecning이란 **기준 전극(reference electrode)를 다른 전극으로 재설정**하는 것을 말합니다. 

만약 EEG를 측정하는 동안의 online reference가 Fz였을 때, 모든 채널에서의 전압(voltage)는 다음과 같습니다.

$$\begin{align}
  \text{v_chan} = \text{efield_chan - efield_fz} \\
  \text{v_fz} = \text{efield_fz - efield_fz} = 0
\end{align}$$

동시에 online reference로 사용한 Fz의 전압은 0이 됩니다. EEG 측정이 끝나고 offline으로 referece 채널을 변경해야 하는 경우가 생길 수 있습니다.
이를 re-referencing이라고 하며, 모든 채널의 전압에서 바꾸고 싶은 reference의 전압을 빼면됩니다. 예시로 reference 채널을 T8로 변경하면 다음과 같습니다.

$$\begin{split}
  \text{v_chan_newref} &= \text{v_chan - v_t8} \\
                       &= \text{efield_chan - efield_fz - (efield_t8 - efield_fz)} \\
                       &= \text{efield_chan - efield_t8}
\end{split}$$
$$\begin{split}
  \text{v_t8_newref} = \text{v_t8 - v_t8} = 0 
\end{split}$$
$$\begin{split}
  \text{v_fz_newref} &= 0 - \text{v_t8} \\
                     &= - \text{(efield_t8 - efield_fz)} \\
                     &= \text{efield_fz - efield_t8}
\end{split}$$

기존 EEG 신호 v_chan과 re-refencing을 적용한 v_chan_newref를 비교해보면, 단순히 reference 채널만 달라진 것을 확인하실 수 있습니다. 또한, 기존에 reference로 사용한 Fz도 다시 사용할 수 있습니다.

따라서 re-ferencing을 통해 **기존 채널들의 전압은 새로운 reference를 뺀 값**이 되고, **새로운 reference 전압은 0**이 됩니다.

## Re-referencing in MNE-Python

***

MNE를 활용하여 쉽게 re-referencing할 수 있습니다.

예제로 사용할 EEG signal을 plot하면 다음과 같습니다. Fz를 reference electrode로 설정하여 EEG를 기록하였습니다.

~~~python
# Original EEG signal
raw = mne.io.read_raw_brainvision(vhdr_fname, preload=True, verbose=False)
raw.plot(n_channels=15, start=100.0)
~~~

![Original EEG](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/2022-07-12-re-referencing/original.png?raw=true)   
EEG original reference.
{:.figure}

기존의 online reference로 사용한 Fz의 전압을 복원하지 않고 reference 채널을 T8로 바꾸어 re-referencing할 수 있습니다.

~~~python
# Re-referencing: Change the reference electrode(Fz) to T8
raw_newref, ref_data = mne.set_eeg_reference(raw, ref_channels=['T8'])
raw_newref.plot(n_channels=15, start=100.0)
~~~

![Re-referencing without Fz](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/2022-07-12-re-referencing/re_referencing_withoutFz.png?raw=true)   
EEG new reference without restoring the signal at Fz.
{:.figure}

새로운 reference electrode T8이 0이 된 것을 확인하실 수 있습니다. 또한, reference 채널을 T8로 재설정하면서 기존에 reference 채널로 사용한 Fz를 복원할 수도 있습니다.

~~~python
# Re-referencing: Change the reference electrode(Fz) to T8
raw_newref_add = mne.add_reference_channels(raw, ref_channels=['Fz'])
raw_newref_add.set_eeg_reference(ref_channels=['T8'])
raw_newref_add.plot(n_channels=16, start=100.0)
~~~

![Re-referencing with Fz](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/2022-07-12-re-referencing/re_referencing_withFz.png?raw=true)   
EEG new reference with restoring the signal at Fz.
{:.figure}

## References

***

[1] MNE, Setting the EEG reference. [[Online]](https://mne.tools/stable/auto_tutorials/preprocessing/55_setting_eeg_reference.html)

<br>

***

<center>오류나 틀린 부분이 있을 경우 언제든지 댓글 혹은 메일로 지적해주시면 감사하겠습니다.</center>