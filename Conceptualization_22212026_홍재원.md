# Reasoning Guard
딥러닝 기반 미확인 공중 표적 정밀 판별 시스템


![Reasoning Guard Logo](./assets/Logo.png)


22212026 홍재원 hongjaewon0428@gmail.com

## Revision history
| Revision date | Vesrsion # | Description | Author |
| :--- | :--- | :--- | :--- |
| 03/13/2026 | 0.00 | First Concept document | 홍재원 |

## Contents
1. Business purpose 

2. System context diagram 

3. Use case list 

4. Concept of operation  

5. Problem statement 

6. Glossary 

7. References 

<br>

---

## 1. Business Purpose
**1) Project background**


![Reasoning Guard Logo](./assets/그림1.png)
[그림 1] 연합뉴스, 2026.03.03 기준 - 군집 위협 규모에 따른 방어 측 재정적 손해 누적 추이 


 위 [그림 1]에서 볼 수 있듯, 단 100기의 드론이 쇄도하는 군집 공습 상황만 가정해도 방어 측은 순식간에 약 6000억 원의 재정적 손실을 입게 되는 극단적인 소모전 양상에 직면합니다. 이러한 비용 비대칭성 속에서, 최근 중동 지역에서 발생한 대규모 공습 사례처럼, 현대전은 저비용 드론과 고성능 미사일을 혼용하여 방공망을 교란하는 공중 위협이 주된 양상으로 자리 잡았습니다. 특히 수백 개의 드론과 미사일이 동시에 쇄도하는 상황에서 인간 관제병의 판단만으로는 한계가 있으며, 아군기와 적기를 순식간에 오판하여 발생할 수 있는 오발사 위험이 비약적으로 증가하고 있습니다. 단 한 번의 오인 사격은 수십억 원에 달하는 요격 미사일의 낭비를 넘어, 방어막 공백으로 인한 핵심 방공 기지 및 군사 인프라 파괴로 이어져 수천억 원 단위의 천문학적인 2차 피해를 초래합니다. 이에 따라 딥러닝 기반의 CV 기술을 활용하여 공중 객체의 정체를 정밀하게 식별하고, 요격 여부에 대한 ‘시각적 추론’을 제공하는 고신뢰성 SW 엔진이 절대적으로 필요하다고 판단했습니다.

**2) goal**

본 프로젝트의 최종 목표는 PyTorch 기반의 딥러닝 모델을 활용하여 공중 객체 (미사일, 전투기, 드론)을 탐지하고 요격 의사결정을 지원하는 ‘Reasoning Guard : 오발사 방지형 실시간 공중 객체 식별 시스템’을 객체 지향 원칙(OOP)에 입각하여 설계 및 구현하는 것입니다.
* **형상 보존형 전처리 파이프라인** <br>
  Zero-Padding 기법을 도입하여, 고속 비행 객체 고유의 종횡비를 엄격히 보존합니다. 이는 리사이징 과정에서의 형상 왜곡을 방지하여 모델의 특징 추출 성능과 식별 정확도를 극대화하는 핵심 파이프라인이 됩니다.

* **UI 통합형 고신뢰성 의사결정 로직** <br>
  학습된 모델을 바탕으로 요격 대상(전투기, 미사일, 드론)에 포함된다면 Shoot(사격)을, 이외의 객체들이면 Hold(대기)라는 판정 결과를 직관적인 UI를 통해 즉각 제공하는 의사결정 알고리즘을 구축합니다.

* **객체 지향적 모듈화 설계** <br>
  시스템의 유지 보수성과 확장성을 위해 모델 추론, 영상 전처리, UI 제어, 이벤트 로그 관리 등 각 기능을 철저히 모듈화합니다. 

* **대리 객체를 통한 데이터 제약 극복** <br>
  국방 데이터의 보안성 및 가용성을 고려하여, 실제 미사일과 기하학적 형태가 유사한 로켓 데이터셋을 대리 객체로 활용합니다. 이를 통해 데이터 획득의 한계를 극복할 예정입니다.

**3) Target Market**
* 대한민국 국방부 및 방위산업체
* 방공 부대 및 전략 거점
* 미래형 무인 요격 체제

<br>

---

## 2. System Context Diagram Labels


![Reasoning Guard Logo](./assets/그림1.png)
[그림 1] 연합뉴스, 2026.03.03 기준 - 군집 위협 규모에 따른 방어 측 재정적 손해 누적 추이


* Fighter Jet / Drone / Rocket (Proxy) Dataset : 전투기 / 드론 / 로켓(대리) 데이터셋
* Zero-Padding (Shape Preservation) : 제로 패딩 (형상 보존)
* Data Augmentation (Flight Simulation) : 데이터 증강 (비행 환경 모사)
* Normalization & Tensor Conversion : 정규화 및 텐서 변환
* PyTorch Inference Engine : 파이토치 추론 엔진
* Model Learning (Weight Training) : 모델 학습 (가중치 훈련)
* System (Reasoning Guard) : 시스템 (Reasoning Guard)
* Input (Unknown Object Image) : 입력 (미식별 객체 이미지)
* Output (Shoot or Hold Decision) : 출력 (사격 또는 대기 결정)
* User (Operator) : 사용자 (관제병)

<br>

---

## 3. Use case list

### 1) 미식별 공중 객체 이미지 판독 요청
| 구분 | 내용 |
| :--- | :--- |
| **Actor** | Operator (관제병) |
| **Description** | 관제병이 레이더나 카메라망에 포착된 미식별 객체의 영상 프레임 또는 이미지를 시스템에 입력하여 AI 보조 판독을 요청한다. |


### 2) AI 모델 학습 및 가중치 업데이트
| 구분 | 내용 |
| :--- | :--- |
| **Actor** | AI Learning Pipeline (외부 시스템) |
| **Description** | 외부 파이프라인에서 전투기, 드론, 로켓 데이터셋을 학습하여 도출된 최적의 가중치(Weight) 모델을 시스템에 최신화한다. |


### 3) 형상 보존형 데이터 전처리 및 텐서 변환
| 구분 | 내용 |
| :--- | :--- |
| **Actor** | System (Reasoning Guard) |
| **Description** | 입력된 이미지의 비율 왜곡을 막기 위해 제로 패딩(Zero-Padding)을 수행하고, 파이토치 연산을 위한 텐서(Tensor) 형태로 변환한다. |


### 4) 파이토치 기반 객체 식별 및 시각적 추론
| 구분 | 내용 |
| :--- | :--- |
| **Actor** | System (Reasoning Guard) |
| **Description** | 전처리된 데이터를 학습된 모델에 통과시켜, 해당 객체가 전투기/드론/로켓 등 학습된 데이터셋과 일치하는지 확률적으로 추론한다. |


### 5) 사격/대기(Shoot/Hold) 의사결정 판정 결과 출력
| 구분 | 내용 |
| :--- | :--- |
| **Actor** | System (Reasoning Guard) |
| **Description** | 추론 결과가 유효 타격 대상과 일치하면 요격 승인(Shoot), 아니면 대기(Hold) 명령을 도출하여 관제병의 최종 판단을 지원한다. |

<br>

---
## 4. Concept of Operation

### 1) 미식별 공중 객체 이미지 판독 요청
| 구분 | 내용 |
| :--- | :--- |
| **Purpose** | 관제병이 판독을 의뢰할 객체의 원본 이미지 데이터를 시스템에 전달 |
| **Approach** | 사용자가 UI를 통해 실시간 포착된 미식별 객체의 영상 프레임(Input)을 업로드함 |
| **Dynamics** | 관제병이 특정 공중 객체의 정체 확인 및 요격 여부에 대한 AI의 보조적 판단이 필요하다고 느낄 때 |
| **Goals** | AI 파이프라인이 분석할 수 있는 데이터를 시스템에 성공적으로 로드함 |


### 2) AI 모델 학습 및 가중치 업데이트
| 구분 | 내용 |
| :--- | :--- |
| **Purpose** | 최신 학습 데이터를 반영하여 객체 식별의 신뢰도를 유지 |
| **Approach** | 외부 파이프라인에서 추출된 최적의 가중치 파일(Weights)을 시스템에 동기화함 |
| **Dynamics** | 새로운 적군 무기 체계 데이터가 수집되었거나, 모델의 식별 성능 개선이 필요할 때 |
| **Goals** | 시스템이 최신화된 객체 식별 능력을 보유하게 함 |


### 3) 형상 보존형 데이터 전처리 및 텐서 변환
| 구분 | 내용 |
| :--- | :--- |
| **Purpose** | 이미지 왜곡을 방지하고 파이토치(PyTorch) 연산에 최적화된 형태로 변환 |
| **Approach** | Zero-Padding 기법을 적용하고 데이터를 텐서(Tensor) 구조로 변환함 |
| **Dynamics** | 이미지 데이터가 시스템에 입력되어 AI 추론 엔진이 가동되기 직전의 단계 |
| **Goals** | 모델의 특징 추출 성능을 극대화하여 추론의 정확도를 높임 |


### 4) 파이토치 기반 객체 식별 및 시각적 추론
| 구분 | 내용 |
| :--- | :--- |
| **Purpose** | 학습된 데이터를 바탕으로 해당 객체의 종류와 위협 수준을 최종 판별 |
| **Approach** | 전처리된 데이터를 추론 엔진에 통과시켜 각 객체(전투기/드론/미사일 등)에 대한 확률값을 산출함 |
| **Dynamics** | 텐서 데이터가 생성되어 시스템 내 가중치 모델과의 연산이 시작될 때 |
| **Goals** | 해당 객체가 실제 요격이 필요한 '고가치 표적'인지에 대한 신뢰도 높은 데이터를 생성함 |


### 5) 사격/대기(Shoot/Hold) 의사결정 판정 결과 출력
| 구분 | 내용 |
| :--- | :--- |
| **Purpose** | AI 추론 결과를 바탕으로 관제병에게 최종 행동 지침을 제시 |
| **Approach** | 분석 결과가 학습된 유효 타격 대상과 일치할 경우 ‘Shoot’, 아군기나 미끼용 객체일 경우 ‘Hold’를 UI에 시각화하여 출력함 |
| **Dynamics** | AI 엔진의 추론 연산이 완료되어 최종 판정 값이 도출되었을 때 |
| **Goals** | 관제병의 오인 사격을 방지하고 국방 자원의 효율적 운용을 지원함 |

<br>

---

## 5. Problem Statement

**Problem #1: AI 모델의 실시간 추론 신뢰성 확보**
군집 드론과 같이 수많은 객체가 동시다발적으로 진입하는 환경에서, AI가 각 객체를 오인 없이 실시간으로 판별할 수 있는 높은 신뢰도의 추론 알고리즘이 필요함.

**Problem #2: 데이터 비대칭성 및 보안 문제**
실제 미사일 등 군사 기밀 데이터 확보의 어려움으로 인해, 유사한 기하학적 특징을 가진 데이터셋(Proxy Dataset)을 활용하여 학습 성능을 극대화해야 하는 기술적 과제가 존재함.

**Problem #3: 하드웨어 자원 최적화**
파이토치 엔진이 텐서 연산을 수행할 때 발생하는 부하를 최소화하여, 실제 방공 관제 시스템의 저사양 엣지 디바이스에서도 지연 없이 작동해야 함.

<br>

---

## 6. Glossary

| Terms | Description |
| :--- | :--- |
| **Zero-Padding (제로 패딩)** | 객체의 고유한 종횡비(비율) 왜곡을 방지하기 위해 빈 공간을 검은 여백으로 채우는 전처리 기법 |
| **Data Augmentation (데이터 증강)** | 기상의 변화나 다양한 비행 각도 등의 환경을 모사하여 모델의 강건성을 높이는 기법 |
| **Tensor (텐서)** | 파이토치(PyTorch) 딥러닝 엔진이 영상 데이터를 빠르고 효율적으로 연산하기 위해 사용하는 다차원 배열 데이터 구조 |
| **Proxy Dataset (대리 데이터셋)** | 보안상 구하기 힘든 실제 국방 데이터 대신, 형태가 유사한 로켓 등의 이미지로 구성한 학습용 대체 데이터셋 |

<br>

---

## 7. References
* 국방 비용 비대칭성 관련 뉴스 [그림 1] : https://www.yna.co.kr/view/AKR20260303027500009
* 기술적 근거 : 2025.20087v1 (엔비디아 논문) : https://arxiv.org/abs/2505.20087
* 캐글(Kaggle) 데이터셋 출처 <br>
  - 전투기 데이터셋 : https://www.kaggle.com/datasets/jrmymimran/fighterjets <br>
  - 드론 데이터셋 : https://www.kaggle.com/datasets/dasmehdixtr/drone-dataset-uav <br>
  - 로켓(미사일 대리) 데이터셋 : https://www.kaggle.com/datasets/eneskosar19/rocket-dataset-for-image-detection-labelled <br>
