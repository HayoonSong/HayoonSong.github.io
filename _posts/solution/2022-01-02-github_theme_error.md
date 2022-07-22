---
layout: post
title: '[Github Pages Error] The theme could not be found.'
subtitle: Github Pages Error
date: '2022-01-02'
categories:
    - solution
tags:
    - Github Pages
comments: true
published: true

last_modified_at: '2022-01-20'
---

Jekyll 테마 적용 에러에 대해서 살펴봅시다.

## 현상

---

* 깃헙 블로그 **테마 설정**이 안됨
* **Page build failure** 메일을 받음

![Error mail](https://cdn.jsdelivr.net/gh/HayoonSong/Images-for-Github-Pages/solution/01_github_theme_error/error_mail.PNG?raw=true)   
Error mail.
{:.figure}

## Code

---

~~~yml
# Theme
# ---------------------------------------------------------------------------------------

theme: jekyll-theme-hydejack
remote_theme: hydecorp/hydejack@v9
~~~

## Error message

---

github-pages 223 | Error:  The jekyll-theme-hydejack theme could not be found.

![Error pages](https://cdn.jsdelivr.net/gh/HayoonSong/Images-for-Github-Pages/solution/01_github_theme_error/01_error.PNG?raw=true)   
Error pages.
{:.figure}

![Error code](https://cdn.jsdelivr.net/gh/HayoonSong/Images-for-Github-Pages/solution/01_github_theme_error/02_error.PNG?raw=true)   
Error code.
{:.figure}

~~~yml
Run actions/jekyll-build-pages@v1-beta
  Logging at level: debug
Configuration file: /github/workspace/./_config.yml
             Theme: jekyll-theme-hydejack
github-pages 223 | Error:  The jekyll-theme-hydejack theme could not be found.
~~~

## 의역

---

* Jekyll-theme-hydejack 테마를 찾을 수 없음

## 원인

---

* **깃헙의 배포 환경**과 **로컬 환경**이 완전히 **동일하지 않기** 때문임 (참고: [Github Pages와 jekyll, 로컬 환경 이슈](https://fuzzysound.github.io/github-and-jekyll))
  - 로컬 환경의 경우 _config.yml의 theme 옵션을 통해 프로젝트에 설치된 gem으로부터 jekyll 테마를 불러올 수 있음
  - Github Pages의 경우 사용 가능한 theme 옵션 값은 [여기](https://pages.github.com/versions/)에 있는 것으로 한정되며, 이외의 값은 page build warning 메일을 받게 됨 
  - 이를 보완하기 위해 깃헙은 **remote_theme** 옵션을 제공하며, 다른 사람의 깃헙 저장소에 올라와 있는 테마를 사용할 수 있게 함

* **Github Pages에 push할 때는 theme를 주석** 처리하고, **로컬 환경(8000번 포트)에서 확인할 때는 theme를 코드** 처리함
  


## Solution code

---

~~~yml
# Theme
# ---------------------------------------------------------------------------------------

# theme: jekyll-theme-hydejack
remote_theme: hydecorp/hydejack@v9
~~~
