# Original Model Used
이 섹션에서는 프로젝트에 사용된 원본 모델 CatVTON의 주요 설정을 명시한다

'''
title: CatVTON
emoji: 🐈
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.40.0
app_file: app.py
pinned: false
license: cc-by-nc-sa-4.0
'''

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


# dressmyfit(part2)

## 목차
1. 전체 서비스 소개 (Part1 + Part2)
   - 프로젝트 개요
   - 시스템 아키텍처
   - 기능 설명

2. Usage (Part2)
   - Environment
   - 터미널 명령어

---

## 1. 전체 서비스 소개 (Part1 + Part2)
### 프로젝트 개요
DressMyFit은 생성형 AI 기반 2D 이미지 합성 모델을 이용한 가상 피팅 서비스이다. 사용자는 입력한 신체 정보를 바탕으로 생성된 맞춤형 아바타 이미지 또는 업로드한 이미지(Target Image)에 대해 의류 이미지(Inference Image)를 선택/업로드하여 가상 피팅 결과를 확인할 수 있다. 
또한, 이 과정에서 사용자는 의류 피팅 정도를 Fit, Standard, Loose 중에서 선택할 수 있다. 이를 통해 사용자들은 자신에게 맞는 의류를 아바타를 통해 시도하고, 실제로 착용한듯한 경험을 할 수 있다.


### 시스템 아키텍처
![image](https://github.com/user-attachments/assets/c3d84b44-fba2-498d-aee2-bb82dc006e42)

### 주요 기능
1. **사용자 정보 입력**
2. **BMI/키에 따른 맞춤형 아바타 이미지 생성**
3. **사용자 사진 업로드 기능 (Target Image 업로드)**
4. **의류 및 피팅 타입 선택**
5. **의류 사진 업로드 기능 (Inference Image 업로드)**
6. **진행 상황 표시 (Progress Bar)**
7. **피팅 결과 생성 및 출력**

### 서비스 특징
#### (1) [Part2 코드 참고]
사용자의 bmi와 성별, 키 정보를 반영한 아바타를 Unity 엔진에서 Ready Player Me SDK를 이용하여 생성하였다. BMI 4구간과, 성별 2개, 키 5구간(140cm~190cm) 에 따라 총 40개의 남녀 아바타 이미지가 있으며, 사용자가 자신의 키, 몸무게, 성별을 입력하면 해당 정보에 맞는 아바타가 웹 페이지에 디스플레이된다.

#### (2) [Part1 코드 참고]
해당 프로젝트는 CatVTON이라는 의류 합성 생성형 모델을 기반으로 하는 서비스이다. 
기존 모델의 경우, target image가 avatar 이미지인 경우, 기존 코드에서 사용되는 automasker로는 마스크가 정확히 생성되지 않고, 따라서 합셩결과가 정확하지 않은 것을 확인하였다. 

따라서 이러한 경우에 대해서, Automasker가 아닌, 신체 사이즈가 좀 더 반영된 마스크를 이용하고자 하였으며, 이를 위해 SAM 모델을 통해 얻은 이미지에 morphological operation을 수행하여, 마스크 생성 기능을 개선하였다.

#### (3) [Part1 코드 참고]
해당 프로젝트는 CatVTON이라는 의류 합성 생성형 모델을 기반으로 하는 서비스이다. 
기존 모델에는, fitting type을 선택할 수 있는 기능이 없었기에, 해당 기능을 추가하였다.
Fit, Standard, Loose의 세 가지 피팅 타입을 사용자가 설정할 수 있도록 하였고, 그에 따른 가상 피팅 결과도 달라지게 하였다. OpenCV 연산을 통해 마스크가 피팅 타입에 맞춰 자동 조절되어, 각기 다른 스타일의 착용감을 제공한다.


### 서비스 흐름
| ![image](https://github.com/user-attachments/assets/56948894-2bb0-4eaa-85ca-be8c488773c4) | ![image](https://github.com/user-attachments/assets/1c98b7ba-b980-4e63-bed3-cb49d2d05860) |
| ![image](https://github.com/user-attachments/assets/210d524f-eafa-4e3c-944e-7d1fde5b2d4c) | ![image](https://github.com/user-attachments/assets/61da6782-54c2-4588-aae2-833fc3b94d6a) |
| ![image](https://github.com/user-attachments/assets/5c97fa0f-822b-4c2e-bbc6-8b20ef2131a8) | ![image](https://github.com/user-attachments/assets/d7a238d9-2967-4e3e-bfbf-4852353795fd) |


### 결과 화면
#### Case 1
- **조건**: 키 164, 몸무게 65kg/80kg 여성, Upper, Fit Type, 체크무늬 의상 선택
- **결과**:
| ![image](https://github.com/user-attachments/assets/fb576cf7-3389-4119-97b5-d63a4b611879) | ![image](https://github.com/user-attachments/assets/e52ab042-3f58-4abe-a6ef-3e831ce91328) |

#### Case 2
- **조건**: 남성 공항사진 (박서준), Upper, Fit/Loose Type, 패딩 선택
- **결과**:
| ![image](https://github.com/user-attachments/assets/d7d31d3f-2ab5-4bb2-b3ad-4df637a92d9a) | ![image](https://github.com/user-attachments/assets/40876e3f-51f6-4e9c-bb38-06448bcea0b2) |

#### Case 3
- **조건**: 여성 공항사진 (제니), Lower, Fit/Loose Type, 청바지 선택
- **결과**:
| ![image](https://github.com/user-attachments/assets/ed8335a1-71f8-4e63-b6e4-5d0866ad997a) | ![image](https://github.com/user-attachments/assets/45e9c289-9387-4fa6-838d-dd05cac4d7c2) |

## 2. Usage
### Environment
- NVIDIA GeForce RTX 2080 Ti
- Conda 환경 사용
- 필요한 패키지는 `requirements.txt` 파일 참고

### 터미널 명령어
1. **환경 설정**
   ```bash
   conda create -n dressmyfit python=3.10.13
   conda activate dressmyfit
   pip install -r requirements.txt
   ```

2. **로컬 서버 실행**
   ```bash
   uvicorn api2:app --reload --ws websockets --port 3000 --host 0.0.0.0
   ```

