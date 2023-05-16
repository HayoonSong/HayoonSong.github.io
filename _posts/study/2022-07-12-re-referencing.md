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
related_posts:
    - _posts/study/2021-03-05-understanding_eeg.md
comments: true
published: true
last_modified_at: '2023-05-16'
---

MNE에서 Re-referencing하는 방법을 알아보고자 합니다.

* this unordered seed list will be replaced by the toc
{:toc}

## EEG re-referencing   

***

MNE뿐만 아니라 EEGLAB을 포함하여 EEG 신호를 다룰 때 reference electrode을 바꾸는 re-referencing을 쉽게 만날 수 있습니다. Acitve electrode, reference electrode, ground electorde 간의 관계에 대한 설명은 이전 포스팅 [뇌파의 이해](https://hayoonsong.github.io/study/2020-10-26-understanding_eeg/)를 참고해주시길 바랍니다.

Re-referecning이란 **기준 전극(reference electrode)를 다른 전극으로 재설정**하는 것을 말합니다. 

만약 online reference로 T8을 사용하였다면, 모든 채널에서의 전압(voltage)는 다음과 같습니다. 이때, online reference로 사용한 채널의 전압은 0이 됩니다.   

$$\begin{align}
  \text{v_chan} = \text{efield_chan - efield_ref} \\
  \text{v_ref} = \text{efield_ref - efield_ref} = 0
\end{align}$$

EEG 측정이 끝나고 offline으로 referece 채널을 변경해야 하는 경우가 생길 수 있습니다. 
이를 re-referencing이라고 하며, 모든 채널의 전압에서 바꾸고 싶은 reference의 전압을 빼주면 됩니다. 이를 통해 새로운 reference 채널은 0이 됩니다. 또한 기존의 reference 채널은 다른 eeg channel과 동일하게 reference 채널의 efield에서 새로운 reference 채널의 efield를 뺀 값이 됩니다.   

$$\begin{align}
  \text{v_ref}         &= 0 - \text{v_newref} \\
                       &= - \text{(efield_newref - efield_ref)} \\
                       &= \text{efield_ref - efield_newref}
\end{align}$$

구체적으로, EEG 측정 시 online reference 채널로 T8을 사용하였고 Fz로 re-referencing 한다고 가정하면 다음과 같습니다. 

**Online reference**   

$$\begin{align}
  \text{v_chan} = \text{efield_chan - efield_t8} \\
  \text{v_t8} = \text{efield_t8 - efield_t8} = 0 \\
  \text{v_fz} = \text{efield_fz - efield_t8}
\end{align}$$

**Re-reference**   

$$
\begin{split}
  \text{v_chan} &= \text{v_chan - v_fz} \\
                &= \text{efield_chan - efield_t8 - (efield_fz - efield_t8)} \\ 
                &= \text{efield_chan - efield_fz} \\[1em]
  \text{v_t8}   &= 0 - \text{v_fz} \\
                &= - \text{(efield_fz - efield_t8)} \\
                &= \text{efield_t8 - efield_fz} \\[1em]
  \text{v_fz}   &= \text{v_fz - v_fz} \\
                &= 0
\end{split}
$$


Online reference의 v_chan과 Re-reference의 v_chan을 비교해보면, 단순히 reference 채널만 달라진 것을 확인하실 수 있습니다. 또한 기존의 reference 채널인 T8도 다시 사용할 수 있습니다. 따라서 re-ferencing을 통해 **기존 채널들의 전압은 새로운 reference를 뺀 값**이 되고, **새로운 reference 전압은 0**이 됩니다.

## Re-referencing in MNE-Python

***

MNE를 활용하여 쉽게 re-referencing할 수 있습니다.

예제로 사용할 EEG 데이터는 online reference로 T8이 사용되었으며, 31개의 채널로 1시간 동안 측정되었습니다. Re-referecning을 통해 reference electrode를 T8에서 Fz로 바꿔 보겠습니다. Raw 데이터는 다음과 같으며 시각화를 위해 4개의 채널(e.g., Fz, Cz, C3, C4)를 100초만 plotting하였습니다.

~~~python
# Original EEG signal
raw = mne.io.read_raw_brainvision(vhdr_fname, preload=True, verbose=False)
raw_picks = raw.pick_channels(['Fz', 'Cz', 'C3', 'C4'])
print(raw_picks.plot(start=100.0))
~~~

![Original EEG](https://cdn.jsdelivr.net/gh/HayoonSong/Images-for-Github-Pages/study/eeg/2022-07-12-re-referencing/original.png?raw=true)   
EEG original reference.
{:.figure}   

MNE의 `set_eeg_reference`를 통해 re-referencing 할 수 있습니다. `ref_channels`에는 새로운 reference 채널인 Fz를 넣었으며, 하단의 그림을 통해 Fz가 reference 채널이 되면서 0이 된 것을 확인할 수 있습니다.   

~~~python
# Re-referencing: Change the reference electrode(T8) to Fz
raw_newref, ref_data = mne.set_eeg_reference(raw_picks, ref_channels=['Fz'])
print(raw_newref.plot(start=100.0))
~~~   

![Re-referencing](https://cdn.jsdelivr.net/gh/HayoonSong/Images-for-Github-Pages/study/eeg/2022-07-12-re-referencing/re_referencing.png?raw=true)   
EEG new reference without restoring the signal at T8.
{:.figure}   

그러나 기존의 reference 채널이었던 T8은 복원되지 않았으며, 새로운 reference electrode인 Fz가 0인 상태로 데이터에 남아있습니다. 따라서 T8을 복원하고 0이 된 reference channel을 EEG 데이터에서 제거하도록 하겠습니다.

~~~python
# Re-referencing: Change the reference electrode(T8) to Fz
raw_ref = mne.add_reference_channels(raw_picks, ref_channels=['T8'])
raw_newref, ref_data = mne.set_eeg_reference(raw_ref, ref_channels=['Fz'])
print(raw_newref.plot(start=100.0))
~~~

![Re-referencing with T8](https://cdn.jsdelivr.net/gh/HayoonSong/Images-for-Github-Pages/study/eeg/2022-07-12-re-referencing/re_referencing_withT8.png?raw=true)   
EEG new reference with restoring the signal at T8.
{:.figure}

Re-referencing은 EEG data에서 reference electrode를 재설정하는 것으로, 수식을 통해 **신호 측정 후에도 단순히 reference 채널을 변경**할 수 있다는 것을 확인하였습니다. 또한, MNE-Python으로 re-referencing하는 방법과 reference 채널을 복구하는 방법까지 알아보았습니다.  

## References

***

[1] MNE, Setting the EEG reference. [[Online]](https://mne.tools/stable/auto_tutorials/preprocessing/55_setting_eeg_reference.html)

<br>

***

<center>오류나 틀린 부분이 있을 경우 언제든지 댓글 혹은 메일로 지적해주시면 감사하겠습니다.</center>