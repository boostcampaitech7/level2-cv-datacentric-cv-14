# 📋 Project Overview


![project_image](https://github.com/user-attachments/assets/e3da3759-967e-4b0b-907b-ae800795abd3)

카메라로 영수증을 인식할 경우 자동으로 영수증 내용이 입력되는 어플리케이션이 있습니다. 이처럼 OCR (Optical Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나입니다.

OCR은 글자 검출 (text detection), 글자 인식 (text recognition), 정렬기 (Serializer) 등의 모듈로 이루어져 있습니다. 본 대회는 아래와 같은 특징과 제약 사항이 있습니다.

- 본 대회에서는 **다국어 (중국어, 일본어, 태국어, 베트남어)로 작성된 영수증 이미지에 대한 OCR task를 수행**합니다.
- 본 대회에서는 **글자 검출만을 수행**합니다. 즉, 이미지에서 어떤 위치에 글자가 있는지를 예측하는 모델을 제작합니다.
- 본 대회는 제출된 예측 (prediction) 파일로 평가합니다.
- 대회 기간과 task 난이도를 고려하여 코드 작성에 제약사항이 있습니다. 상세 내용은 Data > Baseline Code (베이스라인 코드)에 기술되어 있습니다.
- 모델의 입출력 형식은 다음과 같습니다.
  - 입력 : **8글자가 포함된 JPG 이미지 (학습 총 400장, 테스트 총 120장)**
  - 출력 : **bbox 좌표가 포함된 UFO Format (상세 제출 형식은 Overview > Metric 탭 및 강의 6강 참조)**

<br/>

# 🗃️ Dataset

- 전체 이미지
  - **520 images**
  - train
    - **언어당 100장 총 400images**
  - test
    - **언어당 30장 총 120images**
    - **Public 60장, Private 60장**
- 이미지 크기
  - **다양한 사이즈와 비율로 구성**

<br/>

# 😄 Team Member

<table align="center">
    <tr align="center">
        <td><img src="https://github.com/user-attachments/assets/969d8f90-50b5-40bd-a196-2bbfdd49627f" width="140" height="140"></td>
        <td><img src="https://github.com/user-attachments/assets/1d97183e-dd29-4be9-a52c-5f47e48c7eaa" width="140" height="140"></td>
        <td><img src="https://github.com/user-attachments/assets/b76d9189-515e-4cd0-9130-c70b2a8bcd13" width="140" height="140"></td>
        <td><img src="https://github.com/user-attachments/assets/16c15d86-5692-449f-a1f8-6be93e524aae" width="140" height="140"></td>
        <td><img src="https://github.com/user-attachments/assets/6883c516-ebf8-4868-8509-5841bf139808" width="140" height="140"></td>
        <td><img src="https://github.com/user-attachments/assets/482d51d6-e2f2-40de-baba-1f39f9bb93ff" width="140" height="140"></td>
    </tr>
    <tr align="center">
        <td><a href="https://github.com/kimgeonsu" target="_blank">김건수</a></td>
        <td><a href="https://github.com/202250274" target="_blank">박진영</a></td>
        <td><a href="https://github.com/oweixx" target="_blank">방민혁</a></td>
        <td><a href="https://github.com/lkl4502" target="_blank">오홍석</a></td>
        <td><a href="https://github.com/Soy17" target="_blank">이소영</a></td>
        <td><a href="https://github.com/yejin-s9" target="_blank">이예진</a></td>
    </tr>
    <tr align="center">
        <td>T7103</td>
        <td>T7156</td>
        <td>T7158</td>
        <td>T7208</td>
        <td>T7222</td>
        <td>T7225</td>
    </tr>
</table>

<br/>

## 🏆 Project Result

**_<p align=center>Public Leader Board</p>_**
<img src="https://github.com/user-attachments/assets/3110028c-9b59-472c-bb96-266f5e283e34" alt="Public Leader Board" >

<br>

**_<p align=center>Private Leader Board</p>_**
<img src="https://github.com/user-attachments/assets/c0981cec-4b8b-4344-a202-5e8fe6e42d9e" alt="Private Leader Board" >

<br>

## 🔗 Reference

### [📎 Data Centric Wrap-UP Report](https://drive.google.com/file/d/1A8x_xw0erjPVjA3lfnGIDZlsQig6yQaC/view?usp=sharing)

### [📎 Data Centric Notion](https://violet-join-36b.notion.site/Multilingual-Receipt-OCR-a173306ffcb94da2af29710367a6958b?pvs=4)

<br>

## Commit Convention

1. `Feature` : **새로운 기능 추가**
2. `Fix` : **버그 수정**
3. `Docs` : **문서 수정**
4. `Style` : **코드 포맷팅 → Code Convention**
5. `Refactor` : **코드 리팩토링**
6. `Test` : **테스트 코드**
7. `Comment` : **주석 추가 및 수정**

커밋할 때 헤더에 위 내용을 작성하고 전반적인 내용을 간단하게 작성합니다.

### 예시

- `git commit -m "[#issue] Feature : message content"`

커밋할 때 상세 내용을 작성해야 한다면 아래와 같이 진행합니다.

### 예시

> `git commit`  
> 어떠한 에디터로 진입하게 된 후 아래와 같이 작성합니다.  
> `[header]: 전반적인 내용`  
> . **(한 줄 비워야 함)**  
> 상세 내용

<br/>

## Branch Naming Convention

브랜치를 새롭게 만들 때, 브랜치 이름은 항상 위 `Commit Convention`의 Header와 함께 작성되어야 합니다.

### 예시

- `Feature/~~~`
- `Refactor/~~~`
