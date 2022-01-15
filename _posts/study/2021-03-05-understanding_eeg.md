---
layout: post
title: '[EEG] 뇌파의 이해'
subtitle: EEG
date: '2021-01-05'
categories:
    - study
tags:
    - eeg
comments: true
pusblished: true

last_modified_at: '2021-01-05'
---

뇌파(Electroencephalogram, EEG)에 대해 알아봅니다.

## 뇌파(EEG: Electroencephalogram, 뇌전도)

***

뇌파는 뇌 신경세포들의 전기적 활동을 두피에서 비침습적으로 측정한 생체전기신호로 **`뇌의 활성도를 측정`**하는 지표입니다.

![01_sketch_of_EEG](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/01_understanding_eeg/01_sketch_of_EEG.png?raw=true)

### 뇌파의 생성 원리

***

두피에서 기록된 전기적 활성도는 각 기록 전극 근처의 수천 개의 피라미드 세포(pyramidal cell)로부터의 억제성 또는 흥분성 시냅스후 전위(postynaptic potentals)의 합계를 나타냅니다.

EEG 신호의 세기는 부분적으로 전극 아래에 있는 뉴런들이 얼마나 동시적인 활동을 하는가에 크게 달려있습니다. 일단의 세포들이 동시에 흥분되면, 미약한 신호들이 합쳐져서 하나의 커다란 신호를 일으킵니다. 그러나 각각의 세포가 같은 세기의 흥분을 하지만 시간상 흩어져있을 경우 합쳐진 신호는 미약하고 불규칙적입니다.

![02_synapse](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/01_understanding_eeg/02_synapse.png?raw=true)


### 뇌파의 구성

***

뇌파는 주파수 및 진폭을 통하여 이해할 수 있습니다.

![03_frequency_amplitude](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/01_understanding_eeg/03_frequency_amplitude.PNG?raw=true)

*   주파수(freuqnecy)와 주기(period): 주파수는 1초 동안 반복되는 진동 횟수이며 Hz로 표시합니다. 뇌의 전압변화에 따른 1개의 파(wave)에서 산(crest)과 산, 골(trough)과 골을 연결하여 그 간격에 이르는 시간적 길이(wavelength)를 주기라고 하며 ms로 표기합니다. 주파수와 주기는 역수관계에 있는데 1초를 주기로 나누면 주파수가 구해집니다.
*   진폭(amplitude): 진동의 중심으로부터 극점까지 움직인 거리를 말하며, 뇌파는 약  10~200uV의 진폭을 보입니다.


### 뇌파의 율동성

***

뇌파의 율동(rhythm)은 유사한 모양과 기간을 가진 파형이 규칙적으로 나타나는 것이다. 율동성은 각성상태, 집중상태나 행동상태와 연관이 있다.
리듬들은 주파수 범위에 따라 구분되고, 각 범위는 그리스 문자로 이름지어진다. 

![04_bandpower](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/01_understanding_eeg/04_bandpower.jpg?raw=true)


*   베타 리듬은 14Hz 이상으로서 가장 빠르고, 활성화된 피질을 나타낸다. 
*   알파 리듬은 대략 8~13Hz이고 조용히 각성되어 있는 상태와 연관이 있다. 
*   세타 리든은 4~7Hz이고 어떤 수면상태 동안 발생한다. 
*   델타 리듬은 4Hz 미만으로 아주 느리고 자주 진폭이 크고 깊은 수면의 증명이다.




일반적으로 높은 주파수(frequency)에 낮은 진폭(amplitude)의 리듬은 경계심과 각성 또는 수면 중 꿈상태와 연관이 있다. 낮은 주파수에 높은 진폭의 리듬은 꿈을 꾸지 아는 수면 상태나 병리적인 소수 상태와 관련이 있다. 왜냐하면 피질이 정보를 처리하는 과정에서 가장 활발하게 개입되어 있을 때, 정보가 감각 입력에 의해서 발생되든 내부처리에 의해 발생되든 간에, 피질 뉴런의 활동수준은 상대적으로 높지만 또한 상대적으로 동기화되어있지는 않기 때문이다. 달리 말하면, 각각의 뉴런이나 아주 작은 집단의 뉴런들은 약간씩 다른 복잡한 인지작업 요소들에 적극적으로 개입되어 있다. 뉴런은 신속하게 발화하나 대부분의 이웃 뉴런들과 아주 동시적이진 않다. 이것이 낮은 동시성을 일으켜서, EEG 진폭은 낮고 베타 리듬이 지배적이 된다. 반대로 깊은 수면 중에서 피질 뉴런들은 정보처리과정에 개입하지 않고, 수많은 뉴런들이 공통의 느린 리듬의 입력에 의해 물리적으로 흥분된다. 이 경우에 동시성은 높아지고 EEG 진폭은 커진다.


### 뇌파계

***

뇌파는 두피 전극을 통하여 1) 두피 표면의 전위차를 검출하고 2) 측정된 전기적 신호(μV)를 증폭하며 3) 아날로그 신호를 컴퓨터에 저장하여 분석할 수 있도록(sampling)을 수행하며 이렇게 수정된 신호를 디지털 신호로 변환하는 과정이 이루어진다. 

1) 두피 표면의 전위차


*   Signal eletrode
:
*   Ground electrode
: Singla electrode와 referece electrode 간의 전위차를 구하기 위해 사용됩니다.
*   Reference eletrode






#### Reference

은헌정(2019). 정신건강의학과 의사를 위한 뇌파의 기초. Journal of Korean Neuropsychiatric Association, 58(2), 76-104. [Online]. Available at:
https://jknpa.org/DOIx.php?id=10.4306/jknpa.2019.58.2.76 [Accessed 09 Jan. 2022].

한국보건산업진흥원(2017), 뇌 기능 향상 기술. [Online]. Available at:
https://www.khidi.or.kr/board/view?pageNum=1&rowCnt=20&no1=790&linkId=218521&refMenuId=MENU01524&menuId=MENU01521&maxIndex=00002187499998&minIndex=00002093419998&schType=0&schText=&boardStyle=&categoryId=&continent=&country= [Accessed 07 Jan. 2022].

한국콘텐츠진흥원(2011), BCI(Brain Computer Interface) 기술 동향. [Online]. Available at:
https://www.kocca.kr/cop/bbs/view/B0000144/1313379.do?statisMenuNo=200900 [Accessed 09 Jan. 2022].

파낙토스 [Online]. Availale at:
https://www.panaxtos.com/m_page.php?ps_pname=pro_eeg [Accessed 09 Jan. 2022].

