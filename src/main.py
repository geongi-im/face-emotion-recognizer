import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from dotenv import load_dotenv
from logger_config import get_logger

load_dotenv()

# ============ 시스템 설정 (코드 상수) ============
# 이 값들은 일반적으로 고정되며, 필요시 코드에서 직접 수정
USE_GPU = True  # GPU 사용 여부 (False로 변경하면 CPU만 사용)
ENABLE_VISUALIZATION = True  # 시각화 활성화 (CLI 환경에서는 False로 변경)
BACKBONE = 'MobileNetV3Large'  # 백본 모델
SEED = 42  # 랜덤 시드

# 필수 환경변수 검증
REQUIRED_ENV_VARS = [
    # 데이터셋
    'CLASS_NAMES',
    'DATA_DIR_TRAIN',
    'DATA_DIR_VAL',
    # 이미지 및 배치
    'IMG_SIZE',
    'BATCH_SIZE',
    'VAL_STEPS',
    # 헤드 학습
    'STEPS_PER_EPOCH_HEAD',
    'EPOCHS_HEAD',
    'LEARNING_RATE_HEAD',
    # 파인튜닝
    'STEPS_PER_EPOCH_FINETUNE',
    'EPOCHS_FINETUNE',
    'LEARNING_RATE_FINETUNE',
    # 정규화
    'DROPOUT_RATE',
    'WEIGHT_DECAY',
    # 콜백
    'EARLY_STOPPING_PATIENCE',
    'REDUCE_LR_PATIENCE',
    'REDUCE_LR_FACTOR',
    # 고급 하이퍼파라미터
    'FREEZE_BN_MODE',
    'WARMUP_RATIO',
    'AUTO_STEPS',
    'DISABLE_MIX_IN_LAST_STAGE',
    'EMA_DECAY',
    # 데이터 증강 (MixUp/CutMix)
    'MIXUP_ALPHA',
    'CUTMIX_ALPHA',
    'P_MIXUP',
    'P_CUTMIX',
]

missing_vars = []
for var in REQUIRED_ENV_VARS:
    if os.getenv(var) is None:
        missing_vars.append(var)

if missing_vars:
    print("❌ 필수 환경변수가 누락되었습니다:")
    for var in missing_vars:
        print(f"   - {var}")
    print("\n💡 해결 방법:")
    print("   1. .env.example을 복사하여 .env 파일을 생성하세요")
    print("   2. cp .env.example .env (Linux/Mac) 또는 copy .env.example .env (Windows)")
    print("   3. .env 파일에서 필요한 값들을 설정하세요")
    exit(1)

# 환경변수 로드
AUTO_STEPS = os.getenv('AUTO_STEPS') == 'true'   # 데이터셋 크기로 자동 스텝 계산
FREEZE_BN_MODE = os.getenv('FREEZE_BN_MODE')  # BN 동결 전략: all|s3|adaptive|none
WEIGHT_DECAY = float(os.getenv('WEIGHT_DECAY'))  # L2 정규화 강도
WARMUP_RATIO = float(os.getenv('WARMUP_RATIO'))    # Warmup 비율 (0~1)

# 클래스명 파싱 (쉼표로 구분된 문자열을 리스트로 변환)
CLASS_NAMES_RAW = os.getenv('CLASS_NAMES')
CLASS_NAMES = [name.strip() for name in CLASS_NAMES_RAW.split(',') if name.strip()]

# 클래스명 검증 (logger 생성 이전에 조기 실패)
NUM_CLASSES = len(CLASS_NAMES)
if NUM_CLASSES < 1:
    print("❌ 클래스명이 설정되지 않았습니다.")
    print("💡 .env 파일에서 CLASS_NAMES를 설정하세요.")
    print("   예시: CLASS_NAMES=happy,sad,angry")
    exit(1)

# 클래스명에 빈 문자열 검증 (중복 검증은 이미 위에서 수행됨)
if not all(CLASS_NAMES):
    print("❌ 빈 클래스명이 포함되어 있습니다.")
    print("💡 .env 파일의 CLASS_NAMES에서 빈 값을 제거하세요.")
    exit(1)

if len(CLASS_NAMES) != len(set(CLASS_NAMES)):
    print("❌ 중복된 클래스명이 있습니다.")
    print(f"   설정된 클래스: {CLASS_NAMES}")
    print("💡 .env 파일의 CLASS_NAMES에서 중복을 제거하세요.")
    exit(1)

# 디렉토리 경로 설정
DATA_DIR_TRAIN = os.getenv('DATA_DIR_TRAIN')
DATA_DIR_VAL = os.getenv('DATA_DIR_VAL')

IMG_SIZE = int(os.getenv('IMG_SIZE'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
VAL_STEPS = int(os.getenv('VAL_STEPS'))

STEPS_PER_EPOCH_HEAD = int(os.getenv('STEPS_PER_EPOCH_HEAD'))
EPOCHS_HEAD = int(os.getenv('EPOCHS_HEAD'))
LEARNING_RATE_HEAD = float(os.getenv('LEARNING_RATE_HEAD'))

STEPS_PER_EPOCH_FINETUNE = int(os.getenv('STEPS_PER_EPOCH_FINETUNE'))
EPOCHS_FINETUNE = int(os.getenv('EPOCHS_FINETUNE'))
LEARNING_RATE_FINETUNE = float(os.getenv('LEARNING_RATE_FINETUNE'))

DROPOUT_RATE = float(os.getenv('DROPOUT_RATE'))
EARLY_STOPPING_PATIENCE = int(os.getenv('EARLY_STOPPING_PATIENCE'))
REDUCE_LR_PATIENCE = int(os.getenv('REDUCE_LR_PATIENCE'))
REDUCE_LR_FACTOR = float(os.getenv('REDUCE_LR_FACTOR'))
DISABLE_MIX_IN_LAST_STAGE = os.getenv('DISABLE_MIX_IN_LAST_STAGE') == 'true'

# MixUp/CutMix 하이퍼파라미터
MIXUP_ALPHA = float(os.getenv('MIXUP_ALPHA'))
CUTMIX_ALPHA = float(os.getenv('CUTMIX_ALPHA'))
P_MIXUP = float(os.getenv('P_MIXUP'))
P_CUTMIX = float(os.getenv('P_CUTMIX'))

# Label smoothing: MixUp/CutMix와 중복 규제 방지 위해 기본값 0.0로 조정
# 필요시 환경변수로 덮어쓰기 가능
LABEL_SMOOTHING = os.getenv('LABEL_SMOOTHING')
if LABEL_SMOOTHING is None:
    default_smoothing = 0.0 if (P_MIXUP > 0 or P_CUTMIX > 0) else 0.05
    LABEL_SMOOTHING = default_smoothing
else:
    LABEL_SMOOTHING = float(LABEL_SMOOTHING)
logger = get_logger(__name__)
logger.info(f"🔧 Loss label smoothing: {LABEL_SMOOTHING}")

# 고정 경로 설정
OUT_DIR = "models"

# 시각화 활성화 시에만 matplotlib import
if ENABLE_VISUALIZATION:
    import matplotlib
    matplotlib.use('TkAgg')  # GUI 백엔드 설정 (필요시 변경 가능)
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    
    # matplotlib 한글 폰트 설정 (Noto Sans Korean 사용)
    font_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fonts')
    noto_font_path = os.path.join(font_path, 'NotoSansKR-Regular.ttf')
    
    if os.path.exists(noto_font_path):
        try:
            fm.fontManager.addfont(noto_font_path)
            font_name = fm.FontProperties(fname=noto_font_path).get_name()
            plt.rcParams['font.family'] = [font_name]
            print(f"✅ 한글 폰트 로드 성공: {font_name}")
        except Exception as e:
            print(f"⚠️  Noto Sans Korean 폰트 로드 실패: {e}")
            print("💡 시스템 기본 폰트를 사용합니다.")
            plt.rcParams['font.family'] = ['DejaVu Sans']
    else:
        print("⚠️  Noto Sans Korean 폰트를 찾을 수 없습니다.")
        print("💡 fonts/NotoSansKR-Regular.ttf 파일을 추가하세요.")
        print("   다운로드: https://fonts.google.com/noto/specimen/Noto+Sans+KR")
        plt.rcParams['font.family'] = ['DejaVu Sans']
    
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
    plt.ion()  # 인터랙티브 모드 활성화

# 로깅 설정
def get_backbone_and_preprocess(img_size: int):
    """
    백본 모델과 전처리 함수 반환 (MobileNetV3Large 고정)

    Args:
        img_size: 입력 이미지 크기

    Returns:
        (model_fn, preprocess_fn): 모델 생성 함수와 전처리 함수
    """
    input_shape = (img_size, img_size, 3)

    model_fn = lambda: tf.keras.applications.MobileNetV3Large(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    preprocess_fn = tf.keras.applications.mobilenet_v3.preprocess_input

    return model_fn, preprocess_fn

# 시각화 상태 로깅
if ENABLE_VISUALIZATION:
    logger.info("✅ 실시간 시각화 기능 활성화")
else:
    logger.info("❌ 시각화 기능 비활성화 (CLI 환경 또는 설정에 의해)")

# 하이퍼파라미터 설정 로깅
logger.info("🔧 하이퍼파라미터 설정:")
logger.info(f"   - GPU 사용: {'✅ 활성화' if USE_GPU else '❌ 비활성화'}")
logger.info(f"   - 이미지 크기: {IMG_SIZE}x{IMG_SIZE}")
logger.info(f"   - 배치 크기: {BATCH_SIZE}")
logger.info(f"   - 검증 스텝: {VAL_STEPS}")
logger.info(f"   - 백본: {BACKBONE}")
logger.info(f"   - 헤드 학습 에포크: {EPOCHS_HEAD}")
logger.info(f"   - 헤드 스텝/에포크: {STEPS_PER_EPOCH_HEAD}")
logger.info(f"   - 헤드 학습률: {LEARNING_RATE_HEAD}")
logger.info(f"   - 파인튜닝 에포크: {EPOCHS_FINETUNE}")
logger.info(f"   - 파인튜닝 스텝/에포크: {STEPS_PER_EPOCH_FINETUNE}")
logger.info(f"   - 파인튜닝 학습률: {LEARNING_RATE_FINETUNE}")
logger.info(f"   - 드롭아웃 비율: {DROPOUT_RATE}")
logger.info(f"   - MixUp: p={P_MIXUP}, alpha={MIXUP_ALPHA}")
logger.info(f"   - CutMix: p={P_CUTMIX}, alpha={CUTMIX_ALPHA}")
logger.info("🔄 콜백 설정:")
logger.info(f"   - Early Stopping patience: {EARLY_STOPPING_PATIENCE}")
logger.info(f"   - 학습률 감소 patience: {REDUCE_LR_PATIENCE}")
logger.info(f"   - 학습률 감소 비율: {REDUCE_LR_FACTOR}")

class SimpleRealTimeCallback(tf.keras.callbacks.Callback):
    """간단한 실시간 시각화 콜백 (조건부 실행)"""
    def __init__(self, stage_name="학습"):
        super().__init__()
        self.stage_name = stage_name
        self.losses = []
        self.val_losses = []
        self.accuracies = []
        self.val_accuracies = []
        self.enabled = ENABLE_VISUALIZATION
        self.fig = None
        
        if self.enabled:
            # 그래프 창 초기화
            self.fig = plt.figure(figsize=(12, 5))
            self.fig.canvas.manager.set_window_title(f'훈련 진행 상황 - {stage_name}')
            logger.info(f"🎯 {stage_name} 텍스트 + 그래프 시각화 준비 완료")
        else:
            logger.info(f"📊 {stage_name} 텍스트 진행 상황만 표시")
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
            
        # 데이터 수집 (시각화 여부와 관계없이 항상 수집)
        self.losses.append(logs.get('loss', 0))
        self.val_losses.append(logs.get('val_loss', 0))
        # 정확도 수집 (항상 수집)
        self.accuracies.append(logs.get('accuracy', 0))
        self.val_accuracies.append(logs.get('val_accuracy', 0))
        
        # 텍스트 진행 상황은 항상 표시 (매 에포크)
        self.print_progress(epoch + 1, logs)
        
        # 시각화가 활성화된 경우에만 그래프 표시 (매 에포크)
        if self.enabled:
            self.plot_progress(epoch + 1)
    
    def plot_progress(self, current_epoch):
        """진행 상황 그래프 표시"""
        if not self.enabled or self.fig is None:
            return
            
        # 디버깅: 데이터 확인
        logger.info(f"🔍 [{self.stage_name}] 에포크 {current_epoch} 그래프 데이터:")
        logger.info(f"   - 손실 데이터: {len(self.losses)}개 {self.losses[-3:] if len(self.losses) >= 3 else self.losses}")
        logger.info(f"   - 검증 손실: {len(self.val_losses)}개 {self.val_losses[-3:] if len(self.val_losses) >= 3 else self.val_losses}")
        logger.info(f"   - 정확도: {len(self.accuracies)}개 {self.accuracies[-3:] if len(self.accuracies) >= 3 else self.accuracies}")
        logger.info(f"   - 검증 정확도: {len(self.val_accuracies)}개 {self.val_accuracies[-3:] if len(self.val_accuracies) >= 3 else self.val_accuracies}")
            
        # 기존 그래프 지우기
        self.fig.clear()
        epochs = range(1, current_epoch + 1)
        
        # Loss 서브플롯
        ax1 = self.fig.add_subplot(1, 2, 1)
        
        # 훈련 손실 - 파란색으로 먼저 그리기
        if self.losses:
            ax1.plot(epochs, self.losses, 'b-', label='훈련 손실', linewidth=3, marker='o', markersize=4)
            logger.info(f"✅ 훈련 손실 그래프 그리기 완료: 값 범위 {min(self.losses):.4f}~{max(self.losses):.4f}")
        
        # 검증 손실 - 빨간색으로 나중에 그리기
        if self.val_losses:
            ax1.plot(epochs, self.val_losses, 'r-', label='검증 손실', linewidth=3, marker='s', markersize=4)
            logger.info(f"✅ 검증 손실 그래프 그리기 완료: 값 범위 {min(self.val_losses):.4f}~{max(self.val_losses):.4f}")
        
        ax1.set_title(f'{self.stage_name} - 손실', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 손실 축 범위 자동 조정
        if self.losses or self.val_losses:
            all_losses = []
            if self.losses:
                all_losses.extend(self.losses)
            if self.val_losses:
                all_losses.extend(self.val_losses)
            
            if all_losses:
                min_loss = min(all_losses)
                max_loss = max(all_losses)
                margin = (max_loss - min_loss) * 0.1 if max_loss > min_loss else 0.1
                ax1.set_ylim([max(0, min_loss - margin), max_loss + margin])
        
        # Accuracy 서브플롯
        ax2 = self.fig.add_subplot(1, 2, 2)
        
        # 훈련 정확도 - 파란색으로 먼저 그리기
        if self.accuracies:
            ax2.plot(epochs, self.accuracies, 'b-', label='훈련 정확도', linewidth=3, marker='o', markersize=4)
            logger.info(f"✅ 훈련 정확도 그래프 그리기 완료: 값 범위 {min(self.accuracies):.4f}~{max(self.accuracies):.4f}")
        
        # 검증 정확도 - 빨간색으로 나중에 그리기
        if self.val_accuracies:
            ax2.plot(epochs, self.val_accuracies, 'r-', label='검증 정확도', linewidth=3, marker='s', markersize=4)
            logger.info(f"✅ 검증 정확도 그래프 그리기 완료: 값 범위 {min(self.val_accuracies):.4f}~{max(self.val_accuracies):.4f}")
        
        ax2.set_title(f'{self.stage_name} - 정확도', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)  # 업데이트 간격 늘림
    
    def print_progress(self, current_epoch, logs):
        """CLI 환경을 위한 텍스트 진행 상황 표시"""
        loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        
        progress_msg = f"📈 [{self.stage_name}] 에포크 {current_epoch} | 손실: {loss:.4f}"
        if val_loss > 0:
            progress_msg += f" | 검증 손실: {val_loss:.4f}"
            
        # 정확도 정보 추가 (항상 표시)
        acc = logs.get('accuracy', 0)
        val_acc = logs.get('val_accuracy', 0)
        progress_msg += f" | 정확도: {acc:.4f}"
        if val_acc > 0:
            progress_msg += f" | 검증 정확도: {val_acc:.4f}"
        
        logger.info(progress_msg)

class AugmentSwitchCallback(tf.keras.callbacks.Callback):
    """스테이지별 MixUp/CutMix 적용 여부 토글"""
    def __init__(self, enable: bool, name: str = ""):
        super().__init__()
        self.enable = enable
        self.name = name
    def on_train_begin(self, logs=None):
        try:
            APPLY_MIX.assign(self.enable)
            logger.info(f"🎛️  AugmentSwitch: {self.name} | MixUp/CutMix {'활성화' if self.enable else '비활성화'}")
        except Exception as e:
            logger.warning(f"⚠️  AugmentSwitch 실패: {e}")

class EMACallback(tf.keras.callbacks.Callback):
    """지수이동평균(EMA) 가중치 추적 및 스왑

    스테이지별로 trainable 변수 구성이 바뀌므로(on/off freeze),
    매 학습 시작마다 shadow 변수 목록을 재구성한다.
    """
    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.ema_vars = []
        self.orig_vars = []

    def _rebuild_from_model(self):
        self.ema_vars = [tf.Variable(v, trainable=False, dtype=v.dtype) for v in self.model.trainable_variables]

    def on_train_begin(self, logs=None):
        # 현재 trainable_variables에 맞춰 shadow 변수 재생성 및 초기화
        self._rebuild_from_model()
        for ev, v in zip(self.ema_vars, self.model.trainable_variables):
            ev.assign(v)
        logger.info(f"✅ EMA 초기화 완료 (decay={self.decay}, vars={len(self.ema_vars)})")

    def on_train_batch_end(self, batch, logs=None):
        # 매 배치 업데이트 (현재 trainable 변수를 기준으로 동일 순서 가정)
        for ev, v in zip(self.ema_vars, self.model.trainable_variables):
            ev.assign(self.decay * ev + (1.0 - self.decay) * v)

    def apply_ema_weights(self):
        # 현 시점 trainable 목록과 shadow 개수 불일치 시 재정렬
        if len(self.ema_vars) != len(self.model.trainable_variables):
            self._rebuild_from_model()
        self.orig_vars = [tf.identity(v) for v in self.model.trainable_variables]
        for v, ev in zip(self.model.trainable_variables, self.ema_vars):
            v.assign(ev)
        logger.info("🔄 EMA 가중치 적용")

    def restore_original_weights(self):
        if not self.orig_vars:
            return
        for v, ov in zip(self.model.trainable_variables, self.orig_vars):
            v.assign(ov)
        logger.info("🔙 원래 가중치로 복원")

# ============ 0) 환경: CPU/GPU 설정 ============
def setup_device():
    """CPU/GPU 디바이스 설정"""
    physical_devices = tf.config.list_physical_devices()
    gpu_devices = tf.config.list_physical_devices('GPU')
    
    logger.info(f"사용 가능한 디바이스: {physical_devices}")
    
    if USE_GPU:
        if gpu_devices:
            logger.info(f"🚀 GPU 모드 활성화 - 사용 가능한 GPU: {len(gpu_devices)}개")
            for i, gpu in enumerate(gpu_devices):
                logger.info(f"   GPU {i}: {gpu}")
            
            # GPU 메모리 증가 허용
            try:
                for gpu in gpu_devices:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("✅ GPU 메모리 증가 허용 설정 완료")
            except RuntimeError as e:
                logger.warning(f"⚠️  GPU 메모리 설정 실패 (이미 초기화됨): {e}")
            
            # Mixed Precision 활성화 
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("✅ Mixed Precision (float16) 활성화 - Tensor Core 가속")
            except Exception as e:
                logger.warning(f"⚠️  Mixed Precision 설정 실패: {e}")
            
            # GPU 메모리 제한 설정 (선택사항 - RTX 3060은 12GB)
            # tf.config.experimental.set_memory_limit(gpu_devices[0], 10240)  # 10GB로 제한
            
        else:
            logger.warning("⚠️  GPU 사용이 요청되었지만 GPU를 찾을 수 없습니다. CPU 모드로 전환합니다.")
            tf.config.set_visible_devices([], 'GPU')
    else:
        logger.info("🖥️  CPU 모드 활성화")
        tf.config.set_visible_devices([], 'GPU')
    
    # 최종 사용 디바이스 확인
    available_devices = tf.config.list_logical_devices()
    logger.info(f"실제 사용 디바이스: {available_devices}")

setup_device()

# 랜덤 시드 설정 (SEED는 상단에서 정의됨)
tf.random.set_seed(SEED)
np.random.seed(SEED)
logger.info(f"Random seed set to {SEED}")


os.makedirs(OUT_DIR, exist_ok=True)
logger.info(f"Output directory created: {OUT_DIR}")

# ============ 1) 클래스 정보 로깅 및 검증 ============
logger.info(f"클래스(환경변수): {CLASS_NAMES}, 총 {NUM_CLASSES}개")

if NUM_CLASSES == 1:
    logger.info(f"단일 클래스 분류 모델로 훈련을 진행합니다: {CLASS_NAMES[0]}")
else:
    logger.info(f"{NUM_CLASSES}개 클래스 분류 모델로 훈련을 진행합니다: {CLASS_NAMES}")

# train 폴더 검증
train_folders = sorted([d for d in os.listdir(DATA_DIR_TRAIN) if os.path.isdir(os.path.join(DATA_DIR_TRAIN, d))])
expected_folders = sorted(CLASS_NAMES)

if train_folders != expected_folders:
    logger.error(f"❌ train 폴더 불일치: 실제 {train_folders} ≠ 예상 {expected_folders}")
    raise ValueError(f"Train folder mismatch: {train_folders} != {expected_folders}")

# validation 폴더 검증  
val_folders = sorted([d for d in os.listdir(DATA_DIR_VAL) if os.path.isdir(os.path.join(DATA_DIR_VAL, d))])

if val_folders != expected_folders:
    logger.error(f"❌ validation 폴더 불일치: 실제 {val_folders} ≠ 예상 {expected_folders}")
    raise ValueError(f"Validation folder mismatch: {val_folders} != {expected_folders}")

# 파일 개수 확인
for class_name in CLASS_NAMES:
    train_count = len(os.listdir(os.path.join(DATA_DIR_TRAIN, class_name)))
    val_count = len(os.listdir(os.path.join(DATA_DIR_VAL, class_name)))
    logger.info(f"✅ {class_name}: train {train_count}개, validation {val_count}개")

# ============ 2) 데이터 로드 ============
# 분류 모드: 항상 categorical 라벨 사용
label_mode = "categorical"

# Train 데이터셋 로드
raw_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR_TRAIN,
    labels="inferred",
    label_mode=label_mode,
    color_mode="rgb",               # RGB 입력
    batch_size=None,
    image_size=(IMG_SIZE, IMG_SIZE),
    shuffle=True,
    seed=SEED,
    class_names=CLASS_NAMES,
)

# Validation 데이터셋 로드
raw_val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR_VAL,
    labels="inferred",
    label_mode=label_mode,
    color_mode="rgb",               # RGB 입력
    batch_size=None,
    image_size=(IMG_SIZE, IMG_SIZE),
    shuffle=False,  # validation은 셔플하지 않음
    seed=SEED,
    class_names=CLASS_NAMES,
)

# class_names 확인
assert raw_train_ds.class_names == CLASS_NAMES
assert raw_val_ds.class_names == CLASS_NAMES
logger.info(f"데이터셋 클래스 순서 확인 완료: {raw_train_ds.class_names}")

# ============ 3) 전처리 파이프라인 ============
_backbone_builder, _preprocess_input_fn = get_backbone_and_preprocess(IMG_SIZE)

def preprocess_single(x, y):
    return _preprocess_input_fn(x), y

# MixUp / CutMix 구현 (학습 배치에만 적용)
# 위에서 환경변수로 노출된 값을 사용 (기본값 동일)
def _sample_beta(alpha: float, beta: float, shape):
    a = tf.random.gamma(shape=shape, alpha=alpha, dtype=tf.float32)
    b = tf.random.gamma(shape=shape, alpha=beta, dtype=tf.float32)
    return a / (a + b)

def _mixup(images, labels, alpha=MIXUP_ALPHA):
    bs = tf.shape(images)[0]
    idx = tf.random.shuffle(tf.range(bs))
    images2 = tf.gather(images, idx)
    labels2 = tf.gather(labels, idx)
    lam = _sample_beta(alpha, alpha, [bs, 1, 1, 1])
    images_out = lam * images + (1.0 - lam) * images2
    lam_lbl = tf.reshape(lam, [bs, 1])
    labels_out = lam_lbl * labels + (1.0 - lam_lbl) * labels2
    return images_out, labels_out

def _cutmix(images, labels, alpha=CUTMIX_ALPHA):
    bs = tf.shape(images)[0]
    h = tf.shape(images)[1]
    w = tf.shape(images)[2]
    idx = tf.random.shuffle(tf.range(bs))
    images2 = tf.gather(images, idx)
    labels2 = tf.gather(labels, idx)

    lam = _sample_beta(alpha, alpha, [bs])  # (B,)
    cut_rat = tf.sqrt(1.0 - lam)
    cut_w = tf.cast(tf.cast(w, tf.float32) * cut_rat, tf.int32)
    cut_h = tf.cast(tf.cast(h, tf.float32) * cut_rat, tf.int32)

    # 랜덤 박스 중심
    cx = tf.random.uniform([bs], 0, w, dtype=tf.int32)
    cy = tf.random.uniform([bs], 0, h, dtype=tf.int32)

    x1 = tf.clip_by_value(cx - cut_w // 2, 0, w)
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, h)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, w)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, h)

    # 브로드캐스팅으로 마스크 생성 [B,H,W,1]
    yy = tf.reshape(tf.range(h, dtype=tf.int32), [1, h, 1, 1])
    xx = tf.reshape(tf.range(w, dtype=tf.int32), [1, 1, w, 1])
    y1b = tf.reshape(y1, [bs, 1, 1, 1])
    y2b = tf.reshape(y2, [bs, 1, 1, 1])
    x1b = tf.reshape(x1, [bs, 1, 1, 1])
    x2b = tf.reshape(x2, [bs, 1, 1, 1])
    mask_y = tf.logical_and(yy >= y1b, yy < y2b)
    mask_x = tf.logical_and(xx >= x1b, xx < x2b)
    masks = tf.cast(tf.logical_and(mask_y, mask_x), tf.float32)

    images_out = images * (1.0 - masks) + images2 * masks

    box_areas = tf.cast((y2 - y1) * (x2 - x1), tf.float32)
    lam_adj = 1.0 - (box_areas / tf.cast(h * w, tf.float32))  # 실제 라벨 비율 조정
    lam_adj = tf.reshape(lam_adj, [bs, 1])
    labels_out = lam_adj * labels + (1.0 - lam_adj) * labels2
    return images_out, labels_out

def mix_augment(images, labels):
    r = tf.random.uniform([])
    def do_mixup():
        return _mixup(images, labels)
    def do_cutmix():
        return _cutmix(images, labels)
    return tf.cond(r < P_MIXUP, do_mixup, do_cutmix)

# 스테이지별 MixUp/CutMix 적용 여부를 제어할 플래그 (Callback에서 변경)
APPLY_MIX = tf.Variable(True, dtype=tf.bool)

def maybe_mix_augment(images, labels):
    return tf.cond(APPLY_MIX, lambda: mix_augment(images, labels), lambda: (images, labels))

# 셔플 버퍼 크기 계산
train_count = int(tf.data.experimental.cardinality(raw_train_ds).numpy())
SHUFFLE_BUF = min(train_count, max(2048, min(BATCH_SIZE * 128, 8192)))

# 데이터 셔플 및 파이프라인 구성
AUTOTUNE = tf.data.AUTOTUNE
train_ds = (
    raw_train_ds
    .shuffle(SHUFFLE_BUF, reshuffle_each_iteration=True)   # 이미지 단위 셔플
    .batch(BATCH_SIZE, drop_remainder=True)                # 이후 배치
    .map(preprocess_single, num_parallel_calls=AUTOTUNE)
    .map(maybe_mix_augment, num_parallel_calls=AUTOTUNE)   # 조건부 MixUp/CutMix
    .prefetch(AUTOTUNE)
)
val_ds = (
    raw_val_ds
    .batch(BATCH_SIZE)                                     # 검증은 셔플X
    .map(preprocess_single, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

# 반복 대신 데이터셋 크기 기반 스텝 계산
train_batches = int(tf.data.experimental.cardinality(train_ds).numpy())
val_batches = int(tf.data.experimental.cardinality(val_ds).numpy())
if AUTO_STEPS:
    STEPS_PER_EPOCH_HEAD = train_batches
    STEPS_PER_EPOCH_FINETUNE = train_batches
    val_steps = val_batches if val_batches else VAL_STEPS
    logger.info(f"AUTO_STEPS 활성화: steps_per_epoch(train)={train_batches}, validation_steps={val_batches}")
else:
    val_steps  = min(VAL_STEPS, val_batches) if val_batches else VAL_STEPS

# ============ 4) 모델 구성 ============
base = _backbone_builder()
base.trainable = False  # 1단계: 백본(몸통) 동결, 헤드(머리)만 학습

inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))
# 데이터 증강 레이어(학습시에만 랜덤 동작)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.05, 0.05),
    layers.RandomContrast(0.1),
], name="data_augmentation")

# 헤드 학습 단계에서는 BN을 추론 모드로 고정하여 통계치가 흔들리지 않도록 안정화
x_aug = data_augmentation(inputs)
x = base(x_aug, training=False)
# GAP + GMP 결합 후 LN, 2단 헤드 (Swish)
x_gap = layers.GlobalAveragePooling2D()(x)
x_gmp = layers.GlobalMaxPooling2D()(x)
x = layers.Concatenate()([x_gap, x_gmp])
x = layers.LayerNormalization()(x)
x = layers.Dense(512, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
x = layers.Dropout(max(DROPOUT_RATE, 0.4))(x)
x = layers.Dense(256, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
x = layers.Dropout(max(DROPOUT_RATE, 0.3))(x)

# 분류 모드: softmax 출력 (Mixed Precision 사용 시 float32 출력 유지)
if USE_GPU and tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype='float32')(x)
    logger.info(f"분류 모델 구성: {NUM_CLASSES}클래스 출력 (Mixed Precision - float32 출력)")
else:
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    logger.info(f"분류 모델 구성: {NUM_CLASSES}클래스 출력")

model = models.Model(inputs, outputs)

# ============ 5) 컴파일(학습 규칙) ============
def make_warmup_cosine_lr(base_lr: float, steps_per_epoch: int, epochs: int, warmup_ratio: float = 0.1):
    total_steps = max(1, steps_per_epoch * max(1, epochs))
    warmup_steps = int(total_steps * max(0.0, min(1.0, warmup_ratio)))
    cosine_steps = max(1, total_steps - warmup_steps)

    cosine = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=base_lr, decay_steps=cosine_steps)

    class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, warmup_steps, base_lr, cosine):
            self.warmup_steps = warmup_steps
            self.base_lr = base_lr
            self.cosine = cosine
        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            if self.warmup_steps > 0:
                warmup_lr = self.base_lr * (step / float(self.warmup_steps))
                decay_lr = self.cosine(tf.maximum(0.0, step - self.warmup_steps))
                return tf.where(step < self.warmup_steps, warmup_lr, decay_lr)
            else:
                return self.cosine(step)
    return WarmupCosine(warmup_steps, base_lr, cosine)

lr_head = make_warmup_cosine_lr(LEARNING_RATE_HEAD, STEPS_PER_EPOCH_HEAD, EPOCHS_HEAD, WARMUP_RATIO)
optimizer_head = tf.keras.optimizers.AdamW(learning_rate=lr_head, weight_decay=WEIGHT_DECAY, clipnorm=1.0)

model.compile(
    optimizer=optimizer_head,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=["accuracy"],
)
logger.info("분류 모델로 컴파일 완료")

# 전역 베스트(모든 스테이지 공용) 체크포인트 - val_loss 기준
best_overall_path = os.path.join(OUT_DIR, "best_overall.weights.h5")
best_overall_ckpt = ModelCheckpoint(
    filepath=best_overall_path,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    save_weights_only=True,
)

ema_callback = EMACallback(decay=float(os.getenv('EMA_DECAY')))

# 체크포인트(헤드 단계): 가중치만 HDF5로 저장 → TF2.13 호환 안전
callbacks_head = [
    EarlyStopping(monitor="val_accuracy", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True),
    ModelCheckpoint(
        filepath=os.path.join(OUT_DIR, "best_head.weights.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    ),
    best_overall_ckpt,
    SimpleRealTimeCallback("헤드 학습"),  # 간단한 실시간 시각화
    AugmentSwitchCallback(True, name="Head"),
    ema_callback,
]

logger.info("=== [1단계] 헤드 학습 시작 ===")
def compute_class_weights(data_dir, class_names):
    counts = []
    for name in class_names:
        counts.append(len(os.listdir(os.path.join(data_dir, name))))
    total = sum(counts)
    weights = {}
    for idx, c in enumerate(counts):
        weights[idx] = total / (len(counts) * max(1, c))
    logger.info(f"클래스 가중치: {weights}")
    return weights

class_weights = compute_class_weights(DATA_DIR_TRAIN, CLASS_NAMES)

history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=STEPS_PER_EPOCH_HEAD,
    validation_steps=val_steps,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks_head,
    class_weight=class_weights,
)
logger.info("헤드 학습 완료!")

# ============ 6) 파인튜닝: 단계적 언프로즌(3단계) ============
def freeze_bn(backbone):
    """Backbone 내 BatchNormalization 레이어만 동결(trainable=False)"""
    cnt = 0
    for l in backbone.layers:
        if isinstance(l, tf.keras.layers.BatchNormalization):
            l.trainable = False
            cnt += 1
    return cnt
    
def set_unfreeze_ratio(backbone, ratio: float):
    """모델 뒤쪽 ratio(0~1) 만큼만 trainable=True 로 설정 (예: 0.3 → 뒤 30%)"""
    n = len(backbone.layers)
    cut = int(n * (1.0 - ratio))  # cut 이전: 동결(False), cut 이후: 학습(True)
    for i, l in enumerate(backbone.layers):
        l.trainable = (i >= cut)

def compile_for_ft(lr: float, steps_per_epoch: int, epochs: int):
    """trainable 변경 후 재-컴파일 (필수)"""
    lr_schedule = make_warmup_cosine_lr(lr, steps_per_epoch, epochs, WARMUP_RATIO)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"],
    )

def make_ft_callbacks(stage_name: str, enable_mix: bool = True):
    return [
        EarlyStopping(monitor="val_accuracy",
                      patience=EARLY_STOPPING_PATIENCE,
                      restore_best_weights=True),
        ModelCheckpoint(
            filepath=os.path.join(OUT_DIR, f"best_finetune_{stage_name}.weights.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        ),
        best_overall_ckpt,
        SimpleRealTimeCallback(f"파인튜닝-{stage_name}"),
        AugmentSwitchCallback(enable_mix, name=stage_name),
        ema_callback,
    ]

# 총 FT 에포크를 3등분 (예: 42 → 14/14/14)
ft_total = EPOCHS_FINETUNE
stage_epochs = [ft_total // 3, ft_total // 3, ft_total - 2 * (ft_total // 3)]  # 합=ft_total

# 스테이지별 언프로즌 비율 & 러닝레이트 (감쇠 완화: 1.0 → 0.75 → 0.5)
ratios = [0.50, 0.75, 1.00]  # 권장: 뒤 50% → 75% → 전층
lr1 = LEARNING_RATE_FINETUNE
lr2 = max(1e-6, LEARNING_RATE_FINETUNE*0.75)
lr3 = max(5e-7, LEARNING_RATE_FINETUNE*0.5)
lrs = [lr1, lr2, lr3]

logger.info("=== [2단계] 파인튜닝(단계적 언프로즌) 시작 ===")
for idx, (ratio, epochs, lr) in enumerate(zip(ratios, stage_epochs, lrs), start=1):
    stage_name = f"s{idx}_ratio{int(ratio*100)}"
    logger.info(f"▶ Stage {idx}: 뒤 {int(ratio*100)}% 언프로즌 | epochs={epochs}, lr={lr:g}")

    # 1) 언프로즌 범위 설정
    base.trainable = True
    set_unfreeze_ratio(base, ratio)

    # BN 정책 적용
    if FREEZE_BN_MODE == 'all':
        frozen = freeze_bn(base)
        logger.info(f"[{stage_name}] BN policy=all, frozen: {frozen}")
    elif FREEZE_BN_MODE == 's3':
        if idx < 3:
            frozen = freeze_bn(base)
            logger.info(f"[{stage_name}] BN policy=s3, frozen: {frozen}")
        else:
            logger.info(f"[{stage_name}] BN policy=s3, BN trainable")
    elif FREEZE_BN_MODE == 'adaptive':
        if idx == 1:
            frozen = freeze_bn(base)
            logger.info(f"[{stage_name}] BN policy=adaptive(s1 freeze), frozen: {frozen}")
        elif idx == 2:
            # s2: 마지막 20% 레이어의 BN만 학습 허용
            n = len(base.layers)
            cut = int(n * 0.8)
            frozen = 0
            for i, l in enumerate(base.layers):
                if isinstance(l, tf.keras.layers.BatchNormalization):
                    l.trainable = (i >= cut)
                    if not l.trainable:
                        frozen += 1
            logger.info(f"[{stage_name}] BN policy=adaptive(s2 partial unfreeze), frozen: {frozen}")
        else:
            logger.info(f"[{stage_name}] BN policy=adaptive(s3 all BN trainable)")
    else:
        logger.info(f"[{stage_name}] BN policy=none, BN trainable")

    # 2) 재-컴파일 (trainable 변경 후 반드시 필요)
    compile_for_ft(lr, STEPS_PER_EPOCH_FINETUNE, epochs)

    # 3) 스테이지 전용 콜백
    enable_mix = not (DISABLE_MIX_IN_LAST_STAGE and idx == 3)
    callbacks_stage = make_ft_callbacks(stage_name, enable_mix=enable_mix)

    # 4) 학습
    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=STEPS_PER_EPOCH_FINETUNE,
        validation_steps=val_steps,
        epochs=epochs,
        callbacks=callbacks_stage,
        class_weight=class_weights,
    )

logger.info("파인튜닝(단계적 언프로즌) 완료!")

# ============ 7) 평가 ============
# 전역 베스트로 복원 후 평가 (s1/s2가 더 좋았던 경우도 자동 반영)
if os.path.exists(best_overall_path):
    model.load_weights(best_overall_path)
    logger.info("✅ 전역 베스트 가중치로 복원 완료 (monitor=val_loss)")

val_loss, val_acc = model.evaluate(val_ds, verbose=0)
logger.info(f"최종 검증 정확도: {val_acc:.4f}")

# 간단 TTA(Flip) 평가
def evaluate_tta_flip(model, dataset):
    total = 0
    correct = 0
    for xs, ys in dataset:
        probs1 = model.predict(xs, verbose=0)
        probs2 = model.predict(tf.image.flip_left_right(xs), verbose=0)
        probs = (probs1 + probs2) / 2.0
        preds = np.argmax(probs, axis=1)
        trues = np.argmax(ys.numpy(), axis=1)
        correct += np.sum(preds == trues)
        total += xs.shape[0]
    return correct / max(1, total)

tta_acc = evaluate_tta_flip(model, val_ds)
logger.info(f"TTA(Flip) 검증 정확도: {tta_acc:.4f}")

# EMA 가중치로 재평가 후 더 나은 쪽 유지
try:
    ema_callback.apply_ema_weights()
    val_loss_ema, val_acc_ema = model.evaluate(val_ds, verbose=0)
    logger.info(f"EMA 가중치 검증 정확도: {val_acc_ema:.4f}")
    if val_acc_ema >= val_acc:
        logger.info("📌 EMA 가중치를 최종 가중치로 유지합니다.")
    else:
        ema_callback.restore_original_weights()
except Exception as e:
    logger.warning(f"⚠️  EMA 평가 중 오류: {e}")

# ============ 8) 최종 저장 ============
model_name = f"emotion_classifier_{NUM_CLASSES}_classes"

# (A) TensorFlow SavedModel(폴더)
savedmodel_path = os.path.join(OUT_DIR, f"{model_name}_savedmodel")
model.export(savedmodel_path)
logger.info(f"SavedModel 저장 완료: {savedmodel_path}")

# (B) HDF5 단일 파일
h5_path = os.path.join(OUT_DIR, f"{model_name}.h5")
# 커스텀 학습률 스케줄(WarmupCosine)이 포함된 옵티마이저는 직렬화가 어려우므로
# 옵티마이저 제외 저장으로 호환성 확보
model.save(h5_path, include_optimizer=False)
logger.info(f"HDF5 저장 완료: {h5_path}")

# ============ 빠른 예측 데모 ============
# 검증셋은 셔플하지 않으므로, 시각화/점검용 배치는 랜덤 샘플로 구성
def take_random_val_batch(dataset, batch_size, seed=SEED):
    return dataset.unbatch().shuffle(8192, seed=seed, reshuffle_each_iteration=True).batch(batch_size).take(1)

for xs, ys in take_random_val_batch(val_ds, BATCH_SIZE):
    probs = model.predict(xs, verbose=0)   # (B, NUM_CLASSES)
    preds = np.argmax(probs, axis=1)
    trues = np.argmax(ys.numpy(), axis=1)
    logger.info("샘플 예측:")
    logger.info(f"예측: {[CLASS_NAMES[i] for i in preds]}")
    logger.info(f"정답: {[CLASS_NAMES[i] for i in trues]}")
    break

logger.info("모델 훈련 및 저장 완료!")

# matplotlib 시각화 마무리 (시각화가 활성화된 경우에만)
if ENABLE_VISUALIZATION:
    logger.info("✅ 실시간 시각화 완료!")
    logger.info("💡 그래프 창을 닫으려면 창을 직접 닫으세요.")
    
    # 그래프 창이 열려있는 동안 프로그램 유지
    try:
        plt.show(block=True)  # 그래프 창이 닫힐 때까지 대기
    except KeyboardInterrupt:
        logger.info("⌨️  사용자에 의해 프로그램이 종료됩니다.")
    except Exception as e:
        logger.warning(f"⚠️  시각화 종료 중 오류: {e}")
    finally:
        plt.close('all')  # 모든 그래프 창 닫기
        logger.info("🔚 시각화 종료")
else:
    logger.info("📊 CLI 모드 학습 완료!")
