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
pusblished: true

last_modified_at: '2022-01-09'
---

Jekyll 테마 적용 에러에 대해서 살펴봅시다.

## Code

'# Theme
# ---------------------------------------------------------------------------------------

theme: jekyll-theme-hydejack
remote_theme: hydecorp/hydejack@v9'

## Error message

github-pages 223 | Error:  The jekyll-theme-hydejack theme could not be found.

![01_error](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/solution/01_github_theme_error/01_error.PNG?raw=true)

![02_error](https://github.com/HayoonSong/Images-for-Github-Pages/blob/main/solution/01_github_theme_error/02_error.PNG?raw=true)

---

Run actions/jekyll-build-pages@v1-beta
  Logging at level: debug
Configuration file: /github/workspace/./_config.yml
             Theme: jekyll-theme-hydejack
github-pages 223 | Error:  The jekyll-theme-hydejack theme could not be found.

## 의역

*   Jekyll-theme-hydejack 테마를 찾을 수 없음

## 원인


## Solution code

'# Theme'
'# ---------------------------------------------------------------------------------------'

'# theme: jekyll-theme-hydejack'
remote_theme: hydecorp/hydejack@v9
