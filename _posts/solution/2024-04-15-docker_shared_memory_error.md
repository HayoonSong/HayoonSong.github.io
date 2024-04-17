---
layout: post
title: '[Docker Error] ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).'
subtitle: 공유 메모리 부족 문제 해결하기
date: '2024-04-17'
categories:
    - solution
tags:
    - Docker
comments: true
published: true

last_modified_at: '2024-04-17'
---

Jekyll 테마 적용 에러에 대해서 살펴봅시다.

## 현상

---

* Docker 컨테이너에서 이미지 학습이 안됨
* YOLOv8n 모델 및 Pytorch를 사용함

## Error message

~~~bash
ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
OSError: [Errno 28] No space left on device
RuntimeError: unable to write to file </torch_12969_1296138456_0>: No space left on device (28)
~~~

![Error message 1](https://cdn.jsdelivr.net/gh/HayoonSong/Images-for-Github-Pages/solution/02_docker_shared_memory_error/01_error1.png?raw=true)
Error message
{:.figure}

![Error message 2](https://cdn.jsdelivr.net/gh/HayoonSong/Images-for-Github-Pages/solution/02_docker_shared_memory_error/01_error2.png?raw=true)
Error message
{:.figure}

![Error message 3](https://cdn.jsdelivr.net/gh/HayoonSong/Images-for-Github-Pages/solution/02_docker_shared_memory_error/01_error3.png?raw=true)
Error message
{:.figure}

![Error message 4](https://cdn.jsdelivr.net/gh/HayoonSong/Images-for-Github-Pages/solution/02_docker_shared_memory_error/01_error4.png?raw=true)
Error message
{:.figure}

## 의역

---

* No space left on device 디스크 공간이 부족함
* insufficient shared memory (shm) 공유 메모리 영역이 부족함

## 원인

---

* **디스크 공간 부족**: Docker 이미지, 컨테이너, 볼륨이 누적되어 호스트 시스템의 디스크 공간을 모두 소비했음
* **공유 메모리 부족**: 컨테이너 내부에서 실행되는 프로세스가 요구하는 공유 메모리 양이 할당된 양보다 많음

![Docker state](https://cdn.jsdelivr.net/gh/HayoonSong/Images-for-Github-Pages/solution/02_docker_shared_memory_error/05_docker_state.png?raw=true)
Error message
{:.figure}


## 해결 방안

---

1. ~~불필요한 Docker 리소스 정리~~ `&larr;` 리소스 확인 후 안전한 방법 선택

2. 공유 메모리 부족 해결 `&larr;` Docker 컨테이너 실행 시 **공유 메모리 크기 조정**

~~~bash
docker run --shm-size=2G <image_name>
~~~

![Docker state](https://cdn.jsdelivr.net/gh/HayoonSong/Images-for-Github-Pages/solution/02_docker_shared_memory_error/06_docker_state.png?raw=true)
Error message
{:.figure}

***


<br>

***

<center>오류나 틀린 부분이 있을 경우 언제든지 댓글 혹은 메일로 지적해주시면 감사하겠습니다.</center>
