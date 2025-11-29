import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from PIL import Image
import time
import mediapipe as mp
from dotenv import load_dotenv
from logger_config import get_logger

# 로깅 설정
logger = get_logger(__name__)

# .env 로드
load_dotenv()

# 환경에서 테스트 입력 이미지 크기 로드 (훈련과 동일하게 유지)
ENV_IMG_SIZE = int(os.getenv('IMG_SIZE', '224'))

# ============ 설정 (환경변수 또는 클래스 수 기반 자동 설정) ============
# 우선순위: ENV MODEL_PATH > CLASS_NAMES 길이 기반 자동 경로
_raw_classes = os.getenv('CLASS_NAMES', '')
_names = [n.strip() for n in _raw_classes.split(',') if n.strip()]
_num = len(_names) if _names else 2

# MODEL_PATH = os.getenv('MODEL_PATH', f"models/emotion_classifier_{_num}classes.h5")  # 모델 파일 경로 (.h5 또는 SavedModel 폴더)
# TEST_MODE = "webcam"  # 테스트 모드: "webcam", "image", "video"
# INPUT_PATH = ""  # 이미지 또는 동영상 파일 경로 (image/video 모드에서 필요)

# ============ 예시 설정들 (원하는 것을 주석 해제하여 사용) ============
# 웹캠 모드
# MODEL_PATH = "models/emotion_classifier_5classes.h5"
# TEST_MODE = "webcam"
# INPUT_PATH = ""

# 이미지 모드
# MODEL_PATH = "models/emotion_classifier_5classes.h5"
# TEST_MODE = "image"
# INPUT_PATH = "sample/image.jpg"

# 동영상 모드
# MODEL_PATH = "models/emotion_classifier_5classes.h5"
# TEST_MODE = "video"
# INPUT_PATH = "sample/video.mp4"

# SavedModel 사용 예시
MODEL_PATH = "models/emotion_classifier_5_classes_savedmodel"
TEST_MODE = "webcam"
INPUT_PATH = ""
# ============================================================

class EmotionTester:
    def __init__(self, model_path):
        """
        감정 인식 모델 테스터 초기화
        
        Args:
            model_path (str): 모델 파일 경로 (.h5 또는 SavedModel 폴더)
        """
        self.model_path = model_path
        self.model = None
        self.class_names = []
        self.img_size = ENV_IMG_SIZE  # 훈련 시 IMG_SIZE와 일치시킴
        
        # MediaPipe FaceMesh 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,  # 최대 5개 얼굴 검출
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        logger.info("MediaPipe FaceMesh 초기화 완료")
        
        self._load_model()
        self._load_class_names()
        
    def _load_model(self):
        """모델 로드 (SavedModel/TFSMLayer 또는 H5/Keras)"""
        try:
            if os.path.isdir(self.model_path):
                # SavedModel은 tf.saved_model.load로 로딩 후 signature 직접 호출
                sm = tf.saved_model.load(self.model_path)
                sigs = sm.signatures
                if not sigs:
                    raise ValueError("SavedModel에 signatures가 없습니다.")
                call_ep = os.getenv('TF_SERVING_ENDPOINT', 'serving_default')
                if call_ep not in sigs:
                    call_ep = list(sigs.keys())[0]
                    logger.warning(f"요청한 endpoint가 없어 첫 번째 시그니처로 사용: {call_ep}")
                func = sigs[call_ep]
                # 입력 스펙 확인하여 img_size 동기화
                _, input_kwargs = func.structured_input_signature
                if len(input_kwargs) != 1:
                    logger.warning(f"입력이 {len(input_kwargs)}개입니다. 첫 번째 입력만 사용합니다.")
                in_name, in_spec = list(input_kwargs.items())[0]
                if in_spec.shape.rank == 4 and in_spec.shape[1] is not None:
                    expected = int(in_spec.shape[1])
                    if expected != self.img_size:
                        logger.info(f"입력 크기 {self.img_size}→{expected}로 동기화")
                        self.img_size = expected
                self._saved_signature = func
                self._saved_input_name = in_name
                self.model = None
                logger.info(f"SavedModel(signature) 로드 완료: {self.model_path}, endpoint={call_ep}, input={in_name}")
            else:
                # H5 또는 .keras 파일
                try:
                    # Keras 3의 safe_mode로 인한 오류 우회
                    self.model = tf.keras.models.load_model(self.model_path, compile=False, safe_mode=False)
                    logger.info(f"H5/keras 모델 로드 완료: {self.model_path}")
                except TypeError:
                    # TF/Keras 버전에 따라 safe_mode 인자가 없을 수 있음
                    self.model = tf.keras.models.load_model(self.model_path, compile=False)
                    logger.info(f"H5 모델 로드 완료(호환 모드): {self.model_path}")
                except Exception as e_h5:
                    # H5 로드 실패 시, 동일 스템의 SavedModel 폴더를 signature 호출로 폴백 시도
                    base, _ = os.path.splitext(self.model_path)
                    candidate = f"{base}_savedmodel"
                    if os.path.isdir(candidate):
                        logger.warning(f"H5 로드 실패({e_h5}). SavedModel signature로 폴백 시도: {candidate}")
                        sm = tf.saved_model.load(candidate)
                        sigs = sm.signatures
                        if not sigs:
                            raise ValueError("SavedModel에 signatures가 없습니다.")
                        call_ep = os.getenv('TF_SERVING_ENDPOINT', 'serving_default')
                        if call_ep not in sigs:
                            call_ep = list(sigs.keys())[0]
                            logger.warning(f"요청한 endpoint가 없어 첫 번째 시그니처로 사용: {call_ep}")
                        func = sigs[call_ep]
                        _, input_kwargs = func.structured_input_signature
                        in_name, in_spec = list(input_kwargs.items())[0]
                        if in_spec.shape.rank == 4 and in_spec.shape[1] is not None:
                            expected = int(in_spec.shape[1])
                            if expected != self.img_size:
                                logger.info(f"입력 크기 {self.img_size}→{expected}로 동기화")
                                self.img_size = expected
                        self._saved_signature = func
                        self._saved_input_name = in_name
                        self.model = None
                        logger.info(f"SavedModel(signature) 로드 완료(폴백): {candidate}, endpoint={call_ep}, input={in_name}")
                    else:
                        raise e_h5
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            logger.error("모델 경로가 SavedModel 폴더인지, H5 파일인지 확인하세요.")
            raise
    
    def _load_class_names(self):
        """환경변수(CLASS_NAMES)에서 클래스 이름 로드"""
        raw = os.getenv("CLASS_NAMES")
        if not raw:
            logger.error("CLASS_NAMES 환경변수가 설정되지 않았습니다 (.env 확인).")
            self.class_names = ["happy", "sad"]
            logger.warning(f"기본 클래스 이름 사용: {self.class_names}")
            return

        names = [name.strip() for name in raw.split(',') if name.strip()]
        if not names:
            logger.error("CLASS_NAMES 파싱 결과가 비어 있습니다. 예: CLASS_NAMES=happy,sad")
            self.class_names = ["happy", "sad"]
            logger.warning(f"기본 클래스 이름 사용: {self.class_names}")
            return

        # 중복 제거(순서 유지)
        seen = set()
        deduped = []
        for n in names:
            if n not in seen:
                seen.add(n)
                deduped.append(n)

        self.class_names = deduped
        logger.info(f"클래스 이름 로드 완료(환경변수): {self.class_names}")
    
    def detect_faces(self, frame):
        """
        MediaPipe를 사용하여 프레임에서 얼굴 검출
        
        Args:
            frame: OpenCV 이미지 프레임 (BGR)
            
        Returns:
            list: 검출된 얼굴 영역 리스트 [(x, y, w, h), ...]
        """
        # BGR을 RGB로 변환 (MediaPipe는 RGB 사용)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # FaceMesh 처리
        results = self.face_mesh.process(rgb_frame)
        
        face_boxes = []
        
        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            
            for face_landmarks in results.multi_face_landmarks:
                # 얼굴 랜드마크에서 bounding box 계산
                x_coords = [landmark.x * w for landmark in face_landmarks.landmark]
                y_coords = [landmark.y * h for landmark in face_landmarks.landmark]
                
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                # 여백 추가 (얼굴 영역을 조금 더 크게)
                margin = 20
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(w, x_max + margin)
                y_max = min(h, y_max + margin)
                
                width = x_max - x_min
                height = y_max - y_min
                
                face_boxes.append((x_min, y_min, width, height))
        
        return face_boxes
    
    def draw_face_landmarks(self, frame, draw_landmarks=False):
        """
        얼굴 랜드마크를 그리기 (선택사항)
        
        Args:
            frame: OpenCV 이미지 프레임 (BGR)
            draw_landmarks: 랜드마크를 그릴지 여부
            
        Returns:
            frame: 랜드마크가 그려진 프레임
        """
        if not draw_landmarks:
            return frame
            
        # BGR을 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 얼굴 윤곽선만 그리기
                self.mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_CONTOURS,
                    None,
                    self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
        
        return frame
    
    def preprocess_face(self, face_img):
        """
        얼굴 이미지 전처리
        
        Args:
            face_img: 얼굴 이미지 (numpy array)
            
        Returns:
            numpy array: 전처리된 이미지 배치
        """
        # 크기 조정
        face_resized = cv2.resize(face_img, (self.img_size, self.img_size))
        
        # RGB 변환 (OpenCV는 BGR)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # 배치 차원 추가
        face_batch = np.expand_dims(face_rgb, axis=0)
        
        # MobileNetV3 전처리
        face_preprocessed = preprocess_input(face_batch.astype(np.float32))
        
        return face_preprocessed
    
    def predict_emotion(self, face_img):
        """
        얼굴 이미지에서 감정 예측
        
        Args:
            face_img: 얼굴 이미지 (numpy array)
            
        Returns:
            tuple: (예측된 클래스명, 확신도, 모든 클래스 확률)
        """
        try:
            # 전처리
            preprocessed = self.preprocess_face(face_img)
            
            # 예측
            if hasattr(self, '_saved_signature') and self._saved_signature is not None:
                # SavedModel signature 직접 호출
                fn = self._saved_signature
                key = self._saved_input_name
                outputs = fn(**{key: tf.convert_to_tensor(preprocessed)})
                predictions = outputs
            else:
                predictions = self.model.predict(preprocessed, verbose=0)

            # Keras 3 TFSMLayer 사용 시 dict/list 반환 가능성 처리
            if isinstance(predictions, dict):
                # 첫 번째 출력 텐서를 사용
                predictions = next(iter(predictions.values()))
            elif isinstance(predictions, (list, tuple)):
                predictions = predictions[0]
            
            # 특징 추출기 모드인 경우
            if len(predictions.shape) == 2 and predictions.shape[1] != len(self.class_names):
                logger.info("특징 추출기 모드: 분류 결과 없음")
                return "Feature", 1.0, predictions[0]
            
            # 분류 모드인 경우
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            predicted_class = self.class_names[predicted_class_idx]
            
            return predicted_class, confidence, predictions[0]
            
        except Exception as e:
            logger.error(f"감정 예측 실패: {e}")
            return "Error", 0.0, []
    
    def webcam_mode(self):
        """실시간 웹캠 모드"""
        logger.info("웹캠 모드 시작 (ESC 키로 종료)")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.error("웹캠을 열 수 없습니다")
            return
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("프레임을 읽을 수 없습니다")
                    break
                
                # 얼굴 검출
                faces = self.detect_faces(frame)
                
                # 각 얼굴에 대해 감정 예측
                for (x, y, w, h) in faces:
                    # 얼굴 영역 추출
                    face_img = frame[y:y+h, x:x+w]
                    
                    # 감정 예측
                    emotion, confidence, _ = self.predict_emotion(face_img)
                    
                    # 결과 표시
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    label = f"{emotion}: {confidence:.2f}"
                    cv2.putText(frame, label, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
                # 화면에 표시
                cv2.imshow('Emotion Recognition', frame)
                
                # ESC 키로 종료
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("웹캠 모드 종료")
    
    def image_mode(self, image_path):
        """이미지 모드"""
        logger.info(f"이미지 모드 시작: {image_path}")
        
        if not os.path.exists(image_path):
            logger.error(f"이미지 파일이 존재하지 않습니다: {image_path}")
            return
        
        # 이미지 로드
        frame = cv2.imread(image_path)
        if frame is None:
            logger.error(f"이미지를 로드할 수 없습니다: {image_path}")
            return
        
        # 얼굴 검출
        faces = self.detect_faces(frame)
        
        if len(faces) == 0:
            logger.warning("얼굴이 검출되지 않았습니다")
            cv2.imshow('No Face Detected', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return
        
        logger.info(f"{len(faces)}개의 얼굴이 검출되었습니다")
        
        # 각 얼굴에 대해 감정 예측
        for i, (x, y, w, h) in enumerate(faces):
            # 얼굴 영역 추출
            face_img = frame[y:y+h, x:x+w]
            
            # 감정 예측
            emotion, confidence, probabilities = self.predict_emotion(face_img)
            
            logger.info(f"얼굴 {i+1}: {emotion} (확신도: {confidence:.4f})")
            
            # 모든 클래스별 확률 출력
            if len(probabilities) == len(self.class_names):
                for j, class_name in enumerate(self.class_names):
                    logger.info(f"  {class_name}: {probabilities[j]:.4f}")
            
            # 결과 표시
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{emotion}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # 결과 이미지 표시
        cv2.imshow('Emotion Recognition Result', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def video_mode(self, video_path):
        """동영상 모드"""
        logger.info(f"동영상 모드 시작: {video_path}")
        
        if not os.path.exists(video_path):
            logger.error(f"동영상 파일이 존재하지 않습니다: {video_path}")
            return
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"동영상을 열 수 없습니다: {video_path}")
            return
        
        # 동영상 정보
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"동영상 정보 - FPS: {fps}, 총 프레임: {total_frames}")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("동영상 재생 완료")
                    break
                
                frame_count += 1
                
                # 얼굴 검출
                faces = self.detect_faces(frame)
                
                # 각 얼굴에 대해 감정 예측
                for (x, y, w, h) in faces:
                    # 얼굴 영역 추출
                    face_img = frame[y:y+h, x:x+w]
                    
                    # 감정 예측
                    emotion, confidence, _ = self.predict_emotion(face_img)
                    
                    # 결과 표시
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    label = f"{emotion}: {confidence:.2f}"
                    cv2.putText(frame, label, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                # 프레임 정보 표시
                info_text = f"Frame: {frame_count}/{total_frames}"
                cv2.putText(frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # 화면에 표시
                cv2.imshow('Video Emotion Recognition', frame)
                
                # 'q' 키로 종료, 스페이스바로 일시정지
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    cv2.waitKey(0)  # 일시정지
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("동영상 모드 종료")


def main():
    """메인 함수"""
    logger.info(f"감정 인식 모델 테스트 시작")
    logger.info(f"모델 경로: {MODEL_PATH}")
    logger.info(f"테스트 모드: {TEST_MODE}")
    if INPUT_PATH:
        logger.info(f"입력 파일: {INPUT_PATH}")
    
    # 모델 파일 존재 확인
    if not os.path.exists(MODEL_PATH):
        logger.error(f"모델 파일이 존재하지 않습니다: {MODEL_PATH}")
        logger.error("상단의 MODEL_PATH 설정을 확인하세요.")
        return
    
    # 입력 파일 확인 (image/video 모드)
    if TEST_MODE in ["image", "video"] and not INPUT_PATH:
        logger.error(f"{TEST_MODE} 모드에서는 INPUT_PATH가 필요합니다")
        logger.error("상단의 INPUT_PATH 설정을 확인하세요.")
        return
    
    if TEST_MODE in ["image", "video"] and not os.path.exists(INPUT_PATH):
        logger.error(f"입력 파일이 존재하지 않습니다: {INPUT_PATH}")
        logger.error("상단의 INPUT_PATH 설정을 확인하세요.")
        return
    
    try:
        # 테스터 초기화
        tester = EmotionTester(MODEL_PATH)
        
        # 모드별 실행
        if TEST_MODE == "webcam":
            tester.webcam_mode()
        elif TEST_MODE == "image":
            tester.image_mode(INPUT_PATH)
        elif TEST_MODE == "video":
            tester.video_mode(INPUT_PATH)
        else:
            logger.error(f"지원하지 않는 테스트 모드: {TEST_MODE}")
            logger.error("TEST_MODE는 'webcam', 'image', 'video' 중 하나여야 합니다.")
            
    except Exception as e:
        logger.error(f"테스트 실행 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()
