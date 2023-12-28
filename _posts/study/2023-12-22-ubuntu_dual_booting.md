---
layout: post
title: '[Research] 윈도우 환경에서 우분투 22.04 듀얼부팅 세팅하기'
description: >
  윈도우 환경에서 우분투 22.04를 듀얼부팅으로 세팅하는 방법을 소개합니다.
subtitle: 윈도우와 우분투 듀얼부팅
date: '2023-12-22'
categories:
    - study
tags:
    - Ubuntu22.04
comments: true
published: true
last_modified_at: '2023-12-28'
---

본 게시글은 윈도우 환경에서 우분투 22.04를 듀얼부팅으로 세팅하는 것을 목표로 합니다. 
   
이 작업을 통해 **하나의 컴퓨터에서 두 가지 운영체제를 선택하여 사용** 할 수 있게 됩니다. 본 가이드를 따라 듀얼부팅을 설정하실 때, 실수할 경우 PC 데이터가 손실될 수 있으니 **반드시 전체 글을 한 번 읽어보시고 시작**하시는 것을 권장드립니다. 만약 중요한 데이터가 있는 경우, 작업 전에 반드시 백업하시길 바랍니다.

* this unordered seed list will be replaced by the toc
{:toc}

## Overview

***

1. Ubuntu 설치 드라이브 만들기
2. 파티션 축소하기
3. 부팅 우선순위 변경하기
4. Ubuntu 설치하기
   
## 환경

***

* 제품(Model): MSI 노트북(MSI GL75 Leopard 10SDK)
* 프로세서(Processor): i7-10750H CPU
* 메모리(RAM): 16GB
* 기존 OS: Window 11
* 추가 OS: Ubuntu 22.04
   
## Ubuntu 설치 드라이브 만들기

***

Ubuntu 22.04.02 사용
   
## 파티션 축소하기

***

윈도우에서 제공하는 디스크 관리 도구를 통해 현재 운영체제가 설치된 파티션의 일부를 축소할 수 있습니다.
   
디스크 관리 도구는 왼쪽 하단의 윈도우 로고를 우클릭한 다음 디스크 관리를 클릭하여 접근할 수 있습니다.
   
![PC Disk Mangement Process](https://cdn.jsdelivr.net/gh/HayoonSong/Images-for-Github-Pages/study/research/2023-12-22-ubuntu_dual_booting/2_partition_hard_disk/1-1_disk_management.png?raw=true){: width="450px" height="472px"}{:.centered}   
윈도우의 디스크 관리 실행 과정
{:.figure}

![PC Disk Mangement](https://cdn.jsdelivr.net/gh/HayoonSong/Images-for-Github-Pages/study/research/2023-12-22-ubuntu_dual_booting/2_partition_hard_disk/1-2_disk_management.JPG?raw=true){: width="600px" height="470px"}{:.centered}   
디스크 관리 실행
{:.figure}

`디스크 관리`를 열고, 우분투 설치를 위한 공간을 확보하고자 하는 파티션을 우클릭한 후 `볼륨 축소`를 선택합니다. 이 과정에서 중요한 데이터의 손실을 방지하기 위해, 반드시 백업을 진행해두는 것이 좋습니다.

![Free Up Disk Spcae Process](https://cdn.jsdelivr.net/gh/HayoonSong/Images-for-Github-Pages/study/research/2023-12-22-ubuntu_dual_booting/2_partition_hard_disk/1-3_disk_management.JPG?raw=true){: width="600px" height="477px"}{:.centered}   
디스크 관리 축소 과정
{:.figure}

이후 나타나는 창에서 `축소할 공간 입력`란에 우분투 설치를 위해 확보하고자 하는 용량을 입력합니다. 우분투를 설치하는데 필요한 하드 드라이브 공간은 **최소 25 GB(25,000 MB)**입니다. 

![Ubuntu docs](https://cdn.jsdelivr.net/gh/HayoonSong/Images-for-Github-Pages/study/research/2023-12-22-ubuntu_dual_booting/2_partition_hard_disk/1-4_ubuntu_doc.png?raw=true){: width="450px" height="342px"}{:.centered}  
우분투 설치 최소 용량(Source: [Ubuntu](https://ubuntu.com/tutorials/install-ubuntu-desktop#1-overview))
{:.figure}


화면에는 단위가 메가바이트(MB)로 되어있지만 윈도우에서는 MiB의 i를 생략하여 MB로 나타내므로, **메비바이트(MiB) 단위**로 작성합니다. 1 GB = 0.931323 GiB 이므로, 최소 용량 25 GiB를 할당한다고 할 때 **25600 MiB를 입력**하면 됩니다. 예시에서는 노트북 용량이 충분하여 노트북의 디폴트 세팅인 약 278.78GiB(285467MiB)를 우분투 설치를 위해 할당했습니다. 

[GiB to MiB](https://www.dataunitconverter.com/gibibyte-to-mebibyte)에서 Input Gibibyte(GiB)에 원하는 용량(GB)을 입력하면 MiB로 변환해줍니다.

![Free Up Disk Spcae Process](https://cdn.jsdelivr.net/gh/HayoonSong/Images-for-Github-Pages/study/research/2023-12-22-ubuntu_dual_booting/2_partition_hard_disk/1-5_disk_management.JPG?raw=true){: width="600px" height="469px"}{:.centered}   
디스크 관리 축소
{:.figure}

우분투 설치를 위한 공간이 성공적으로 확보되었습니다. 축소된 파티션은 '비할당' 상태로 표시될 것입니다. 우분투 설치 과정에서 이 비할당 공간을 새로운 파티션으로 설정하여 사용할 수 있게 됩니다.

## 부팅 우선순위 변경하기

***

컴퓨터를 재시작하고, BIOS 또는 UEFI 설정에 들어갑니다. 이는 대부분의 컴퓨터에서 특정 키(F2, F10, F12, Del 등)를 컴퓨터 부팅 시 눌러서 할 수 있습니다. 설정 메뉴에서 '부팅 순서' 또는 '부팅 우선순위'를 찾아, USB 또는 DVD 드라이브가 첫 번째로 부팅되도록 변경합니다.

## Ubuntu 설치하기

***

부팅 순서를 변경한 후, 컴퓨터를 재시작합니다. 그러면 우분투 설치 화면이 나타납니다. 화면의 지시에 따라 언어, 키보드 레이아웃, 시간대 등을 설정하고, '다른 것'을 선택해 직접 파티션을 설정합니다. 미리 축소해둔 공간을 선택하고 우분투를 설치합니다.

## References

***

[1] DARK TORNADO, 디스크 파티션 분할하기. [[Online]](https://darktornado.github.io/blog/disk-partition/)   
[1] 감성코딩, 윈도우10, 우분투 18.04 듀얼부팅 설정. [[Online]](https://tlo-developer.tistory.com/96)   


<br>

***

<center>오류나 틀린 부분이 있을 경우 언제든지 댓글 혹은 메일로 지적해주시면 감사하겠습니다.</center>