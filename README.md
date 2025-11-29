# 한국인 얼굴 감정 인식 프로젝트

MobileNetV3Large 기반 딥러닝 모델을 활용한 실시간 얼굴 감정 인식 시스템입니다.

## 데이터셋

**출처**: [AIHub - 한국인 감정인식을 위한 복합 영상](https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%EA%B0%90%EC%A0%95&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&srchDataRealmCode=REALM001&aihubDataSe=data&dataSetSn=82)

**규모**:
- 5개 감정 클래스: `happy`, `sad`, `neutral`, `embarrassed`, `anxiety`
- 클래스당 약 70,000개 이미지
- **총 350,000개 이미지** 사용

**학습 환경**:
- **OS**: Ubuntu
- **GPU**: NVIDIA RTX 3060 12GB
- **Python**: 3.10
- **TensorFlow**: 2.13

## 주요 기능

### 1. 데이터 전처리 ([preprocess.py](src/preprocess.py))
- AIHub 원본 이미지를 MediaPipe를 사용하여 얼굴 영역 검출 및 크롭
- 정사각형 변환 + 여백 추가
- 전처리된 데이터를 `dataset_crop/` 폴더에 저장

### 2. 모델 학습 ([main.py](src/main.py))
- MobileNetV3Large 백본 사용
- 224×224 입력 크기
- 2단계 훈련: 헤드 학습 → 파인튜닝
- MixUp/CutMix 데이터 증강 적용
- EMA (Exponential Moving Average) 가중치 평균화

### 3. 웹 서버 ([server.py](src/server.py))
- FastAPI 기반 실시간 감정 분석 API
- WebSocket을 통한 실시간 영상 처리
- MediaPipe 얼굴 검출
- 다중 얼굴 동시 분석 지원
- 포트: 8001

### 4. 로컬 테스트 ([run_model.py](src/run_model.py))
- 웹캠/이미지/비디오 모드 지원
- MediaPipe 기반 얼굴 검출
- 실시간 감정 분석 결과 표시

## 📁 프로젝트 구조

```
face-emotion-recognizer/
├── src/
│   ├── main.py              # 모델 학습 스크립트
│   ├── server.py            # FastAPI 웹 서버
│   ├── run_model.py         # 로컬 테스트 (웹캠/이미지/비디오)
│   ├── preprocess.py        # 데이터 전처리
│   ├── logger_config.py     # 로깅 설정
│   └── static/              # 웹 UI 파일
│       ├── index.html       # 웹 인터페이스
│       ├── style.css
│       └── script.js
├── dataset_crop/            # 전처리된 데이터셋
│   ├── train/               # 학습 데이터
│   │   ├── happy/
│   │   ├── sad/
│   │   ├── neutral/
│   │   ├── embarrassed/
│   │   └── anxiety/
│   └── validation/          # 검증 데이터 (자동 분할)
├── models/                  # 학습된 모델 저장
│   ├── emotion_classifier_5_classes.h5
│   └── emotion_classifier_5_classes_savedmodel/
├── logs/                    # 로그 파일
│   └── app_YYYY_MM_DD.log
├── .env                     # 환경변수 설정
├── .env.example             # 환경변수 예시 파일
├── requirements.txt         # 필요한 패키지 목록
└── README.md
```

## 빠른 시작

### 1. 환경 설정

```bash
# 1. 저장소 클론
git clone <repository-url>
cd face-emotion-recognizer

# 2. 가상환경 생성 및 활성화
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. 패키지 설치
pip install -r requirements.txt
```

### 2. 환경변수 설정

`.env.example` 파일을 복사하여 `.env` 파일 생성:

```bash
cp .env.example .env  # Linux/Mac
copy .env.example .env  # Windows
```

`.env` 파일을 열어 필요한 값 수정:

```bash
# 데이터셋 설정
CLASS_NAMES=happy,sad,neutral,embarrassed,anxiety
DATA_DIR_TRAIN=dataset_crop/train
DATA_DIR_VAL=dataset_crop/validation

# 이미지 및 배치 설정
IMG_SIZE=224
BATCH_SIZE=32
VAL_STEPS=500

# ... (기타 하이퍼파라미터)
```

**필수 환경변수 (총 29개)**: 자세한 내용은 `.env.example` 파일 참고

### 3. 데이터 전처리

AIHub에서 다운로드한 원본 데이터를 전처리합니다:

```bash
python src/preprocess.py
```

**전처리 과정**:
1. JSON 어노테이션 파일 읽기
2. MediaPipe로 얼굴 영역 검출
3. 정사각형 크롭 + 여백 추가
4. `dataset_crop/train/{감정}/` 폴더에 저장

### 4. 모델 학습

```bash
python src/main.py
```

**학습 과정**:
1. **헤드 학습** (6 에포크): 분류 레이어만 훈련
2. **파인튜닝** (42 에포크): 백본 상위 레이어 언프리즈 후 훈련
3. **모델 저장**: HDF5 (`.h5`) 및 SavedModel 형식으로 저장

**학습 시간 (RTX 3060 12GB 기준)**:
- 헤드 학습: 약 15-20분
- 파인튜닝: 약 1-1.5시간
- **총 소요 시간**: 약 1.5-2시간

## 사용 방법

### 1. 웹 서버 모드 (FastAPI)

```bash
python src/server.py
```

브라우저에서 `http://localhost:8001` 접속

**기능**:
- 실시간 웹캠 영상 처리
- WebSocket 기반 실시간 감정 분석
- 다중 얼굴 동시 검출 및 분석
- 각 얼굴별 감정 확률 표시

### 2. 로컬 테스트 모드

#### 웹캠 모드 (실시간)
```bash
python src/run_model.py
```

#### 이미지 모드
`.env` 파일에서 설정 변경:
```bash
TEST_MODE=image
INPUT_PATH=path/to/image.jpg
```

#### 비디오 모드
`.env` 파일에서 설정 변경:
```bash
TEST_MODE=video
INPUT_PATH=path/to/video.mp4
```

## 모델 아키텍처

### 백본: MobileNetV3Large
- ImageNet 사전 훈련 가중치 사용
- 입력 크기: 224×224×3
- 경량화된 구조로 실시간 처리 가능

### 분류 헤드
```python
GlobalAveragePooling2D()
Dense(256, activation='relu')
Dropout(0.25)
Dense(5, activation='softmax')  # 5개 감정 클래스
```

### 데이터 증강
- **MixUp**: 이미지 혼합 (alpha=0.2, p=0.7)
- **CutMix**: 영역 교체 (alpha=0.2, p=0.3)
- **BatchNormalization 동결**: 안정적인 학습

### 훈련 전략
1. **헤드 학습**:
   - 백본 동결
   - 학습률: 0.0003
   - 에포크: 6

2. **파인튜닝**:
   - 백본 상위 레이어 언프리즈
   - 학습률: 0.00002
   - 에포크: 42
   - EMA 가중치 평균화 적용

3. **콜백**:
   - Early Stopping (patience=8)
   - ReduceLROnPlateau (patience=3, factor=0.5)
   - ModelCheckpoint (val_loss 기준)

## 로깅 시스템

### 단일 통합 로그
- 파일: `logs/app_{날짜}.log`
- 모든 스크립트의 로그를 하나의 파일에 통합
- 파일 소스 자동 포함 (`src.main`, `src.preprocess`, `src.server` 등)

### 로그 예시
```
2025-11-30 14:23:45 - src.main - INFO - 🔧 하이퍼파라미터 설정:
2025-11-30 14:23:45 - src.main - INFO -    - GPU 사용: ✅ 활성화
2025-11-30 14:23:45 - src.main - INFO -    - 백본: MobileNetV3Large
2025-11-30 14:23:45 - src.server - INFO - WebSocket 연결 수락됨
2025-11-30 14:23:46 - src.server - INFO - Frame received. Faces detected: 2
```

## 주요 환경변수 설명

### 데이터셋
- `CLASS_NAMES`: 감정 클래스 (쉼표 구분)
- `DATA_DIR_TRAIN`: 학습 데이터 경로
- `DATA_DIR_VAL`: 검증 데이터 경로

### 학습 하이퍼파라미터
- `IMG_SIZE`: 입력 이미지 크기 (224 권장)
- `BATCH_SIZE`: 배치 크기 (GPU: 32-64, CPU: 16-32)
- `EPOCHS_HEAD`: 헤드 학습 에포크 수
- `EPOCHS_FINETUNE`: 파인튜닝 에포크 수
- `LEARNING_RATE_HEAD`: 헤드 학습률
- `LEARNING_RATE_FINETUNE`: 파인튜닝 학습률

### 정규화
- `DROPOUT_RATE`: 드롭아웃 비율 (0.25 권장)
- `WEIGHT_DECAY`: L2 정규화 강도 (0.00001 권장)
- `FREEZE_BN_MODE`: BN 동결 전략 (all/s3/adaptive/none)

### 데이터 증강
- `MIXUP_ALPHA`: MixUp 알파 값 (0.2 권장)
- `CUTMIX_ALPHA`: CutMix 알파 값 (0.2 권장)
- `P_MIXUP`: MixUp 적용 확률 (0.7 권장)
- `P_CUTMIX`: CutMix 적용 확률 (0.3 권장)

전체 환경변수 목록은 [.env.example](.env.example) 참고

## 기술 스택

### 딥러닝 프레임워크
- TensorFlow 2.13
- Keras

### 얼굴 검출
- MediaPipe Face Mesh
- OpenCV

### 웹 서버
- FastAPI
- WebSocket
- Uvicorn

### 기타
- python-dotenv (환경변수 관리)
- NumPy
- Pillow

## 성능

### RTX 3060 12GB 환경
- **학습 시간**: 약 1.5-2시간 (전체 350,000개 이미지)
- **추론 속도**:
  - 웹캠 실시간: 약 30 FPS
  - 단일 이미지: 약 20ms
- **메모리 사용**: 약 4-6GB VRAM

### 모델 성능
- **정확도**: 데이터셋 및 훈련 결과에 따라 다름
- **모델 크기**: 약 20MB (H5 형식)

## 주의사항

1. **GPU 메모리**: RTX 3060 12GB 이상 권장
2. **데이터셋 경로**: AIHub 데이터를 `preprocess.py`에서 지정한 경로에 배치
3. **환경변수**: `.env` 파일 필수 생성
4. **Python 버전**: 3.10 권장 (3.8 이상 지원)

## 트러블슈팅

### 1. GPU 메모리 부족
```bash
# .env 파일에서 배치 크기 줄이기
BATCH_SIZE=16
```

### 2. MediaPipe 설치 오류
```bash
pip install --upgrade mediapipe
```

### 3. TensorFlow GPU 인식 안됨
```bash
# CUDA 11.8 + cuDNN 8.6 설치 필요
# TensorFlow 2.13은 CUDA 11.8과 호환
```

### 4. 로그 확인
```bash
# 실시간 로그 모니터링
tail -f logs/app_2025_11_30.log  # Linux/Mac
Get-Content logs\app_2025_11_30.log -Wait  # Windows PowerShell
```

---

**개발 환경**: Python 3.10, TensorFlow 2.13, Ubuntu + RTX 3060 12GB
**데이터셋**: AIHub "한국인 감정인식을 위한 복합 영상" (350,000개 이미지)
