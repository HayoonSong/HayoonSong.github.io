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

last_modified_at: '2022-01-17'
---

뇌파(Electroencephalogram, EEG)에 대해 알아봅니다.

## 뇌파(EEG: Electroencephalogram, 뇌전도)

***

뇌파는 두뇌를 구성하는 신경세포들의 전기적 활동을 두피에서 전극을 통해 간접적으로 측정할 수 있는 생체전기신호로 **`뇌의 활성도를 측정`**하는 지표입니다.    

<center><img src="https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/01_understanding_eeg/sketch_of_EEG.png?raw=true" alt="Sketch of EEG" width="600" height="600"></center>


### 뇌파의 생성 원리

***

두피에서 기록된 전기적 활성도는 각 기록 전극 근처의 수천 개의 피라미드 세포(pyramidal cell)에서 발생하는 억제성 또는 흥분성 시냅스후 전위(postynaptic potentals)의 합계를 나타냅니다. EEG 신호의 세기는 부분적으로 전극 아래에 있는 뉴런들이 얼마나 동시적인 활동을 하는가에 크게 달려있습니다. 
*   동기화: 세포들이 동시에 흥분되면, 미약한 신호들이 합쳐져서 하나의 커다란 신호를 일으킵니다. 
*   불규칙적: 각각의 세포가 같은 세기의 흥분을 하지만 시간상 흩어져있을 경우 합쳐진 신호는 미약합니다.

<center><img src="https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/01_understanding_eeg/synapse.png?raw=true" alt="synapse"></center>


### 뇌파의 측정

***

뇌파는 신경전류의 흐름이 두피에 만드는 전위차를 측정하는 것이므로 뇌파의 측정을 위해서는 최소 2개의 전극이 필요합니다. 하나의 전극을 기준 전극(reference electrode) 으로 지정하고 기준 전극과 다른 전극들 사이의 전위차를 측정하는 단극(unipolar) 측정 방식에서는 평균 기준 전극 방식을 이용하면 절대 전위를 근사할 수 있지만, 머리가 완전 구형이며 매우 고밀도로 전극이 되어 있어야 한다는 가정이 필요하기 때문에, 실제로는 절대 전위와 어느 정도 차이가 있습니다. 그럼에도 불구하고 평균 기준을 적용한 전위 데이터는 기준 전극 위치에 독립적인 두피 전위 지도를 그리거나 신호원 영상을 하기 위해 자주 사용됩니다.

*   측정 전극(signal eletrode)
: 실제로 신경 활성이 일어난다고 추측되는 곳 근처에 붙이는 전극입니다.
*   기준 전극(reference eletrode)
: 이상적인 기준 전극은 측정하고자 한느 전극 위치에 대해 비활성화된, 즉 측정하고자 하는 전극 위치의 전기적 필드에 전혀 영향을 받지 않는 위치입니다.
*   접지 전극(ground electrode)
: 모든 전위에 대하여 기준이 되는 전압을 의미하여, 측정 전극과 기준 전극 간의 정확한 전위차를 구하기 위해 필요합니다.

### 뇌파의 구성

***

뇌파는 주파수 및 진폭을 통하여 이해할 수 있습니다.

<center><img src="https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/01_understanding_eeg/frequency_amplitude.PNG?raw=true" alt="frequency"></center>


*   **주파수(freuqnecy)**와 주기(period): 주파수는 1초 동안 반복되는 진동 횟수이며 Hz로 표시합니다. 뇌의 전압변화에 따른 1개의 파(wave)에서 산(crest)과 산, 골(trough)과 골을 연결하여 그 간격에 이르는 시간적 길이(wavelength)를 주기라고 하며 ms로 표기합니다. 주파수와 주기는 역수관계에 있는데 1초를 주기로 나누면 주파수가 구해집니다.
*   **진폭(amplitude)**: 진동의 중심으로부터 극점까지 움직인 거리를 말하며, 뇌파는 약  10~200uV의 진폭을 보입니다.


### 뇌파의 양상

***

뇌파의 분류방법으로 신호의 진동수(주파수)에 따라 분류하는 **파워 스펙트럼 분류**를 가장 많이 사용합니다. 파워 스펙트럼은 측정되는 뇌파신호를 특정한 주파수별 단순 신호들의 선형적 합산으로 보고, 이 신호를 각각의 주파수 성분별로 분해하여 그 크기(전력치)를 표시한 것입니다.

<center><img src="https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/study/eeg/01_understanding_eeg/bandpower.jpg?raw=true" alt="bandpower"></center>


### References

***

*   은헌정(2019). 정신건강의학과 의사를 위한 뇌파의 기초. Journal of Korean Neuropsychiatric Association, 58(2), 76-104. [Online]. Available at: <https://jknpa.org/DOIx.php?id=10.4306/jknpa.2019.58.2.76> [Accessed 09 Jan. 2022].

*   김도원 & 김명선(2017). 뇌파의 이해와 응용. 학지사

*   김도영, 이재호, 박문호, 최윤호, & 박윤옥. (2017). 뇌파신호 및 응용 기술 동향. [ETRI] 전자통신동향분석, 32(2), 0-0. [Online]. Available at: <https://www.koreascience.or.kr/article/JAKO201752055796148.pdf> [Accessed 16 Jan. 2022]

*   한국보건산업진흥원(2017), 뇌 기능 향상 기술. [Online]. Available at: <https://www.khidi.or.kr/board/view?pageNum=1&rowCnt=20&no1=790&linkId=218521&refMenuId=MENU01524&menuId=MENU01521&maxIndex=00002187499998&minIndex=00002093419998&schType=0&schText=&boardStyle=&categoryId=&continent=&country=> [Accessed 07 Jan. 2022].

*   한국콘텐츠진흥원(2011), BCI(Brain Computer Interface) 기술 동향. [Online]. Available at: <https://www.kocca.kr/cop/bbs/view/B0000144/1313379.do?statisMenuNo=200900> [Accessed 09 Jan. 2022].

*   파낙토스 [Online]. Availale at: <https://www.panaxtos.com/m_page.php?ps_pname=pro_eeg> [Accessed 09 Jan. 2022].

