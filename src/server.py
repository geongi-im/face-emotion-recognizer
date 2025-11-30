import os
import sys
# Add project root to sys.path to allow running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import base64
import json
import uvicorn
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import mediapipe as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# .env 로드
load_dotenv()

# 환경변수에서 설정 로드
ENV_IMG_SIZE = int(os.getenv('IMG_SIZE', '224'))
SERVER_PORT = int(os.getenv('SERVER_PORT', '8001'))
SERVER_HOST = os.getenv('SERVER_HOST', '0.0.0.0')
MODEL_PATH = "models/emotion_classifier_5_classes_savedmodel"


# ============ EmotionTester 클래스 (웹 서버용) ============
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
        self.img_size = ENV_IMG_SIZE

        # MediaPipe FaceMesh 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

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
                    print(f"요청한 endpoint가 없어 첫 번째 시그니처로 사용: {call_ep}")
                func = sigs[call_ep]
                # 입력 스펙 확인하여 img_size 동기화
                _, input_kwargs = func.structured_input_signature
                if len(input_kwargs) != 1:
                    print(f"입력이 {len(input_kwargs)}개입니다. 첫 번째 입력만 사용합니다.")
                in_name, in_spec = list(input_kwargs.items())[0]
                if in_spec.shape.rank == 4 and in_spec.shape[1] is not None:
                    expected = int(in_spec.shape[1])
                    if expected != self.img_size:
                        print(f"입력 크기 {self.img_size}→{expected}로 동기화")
                        self.img_size = expected
                self._saved_signature = func
                self._saved_input_name = in_name
                self.model = None
                print(f"SavedModel(signature) 로드 완료: {self.model_path}, endpoint={call_ep}, input={in_name}")
            else:
                # H5 또는 .keras 파일
                try:
                    self.model = tf.keras.models.load_model(self.model_path, compile=False, safe_mode=False)
                    print(f"H5/keras 모델 로드 완료: {self.model_path}")
                except TypeError:
                    self.model = tf.keras.models.load_model(self.model_path, compile=False)
                    print(f"H5 모델 로드 완료(호환 모드): {self.model_path}")
                except Exception as e_h5:
                    # H5 로드 실패 시, 동일 스템의 SavedModel 폴더를 signature 호출로 폴백 시도
                    base, _ = os.path.splitext(self.model_path)
                    candidate = f"{base}_savedmodel"
                    if os.path.isdir(candidate):
                        print(f"H5 로드 실패({e_h5}). SavedModel signature로 폴백 시도: {candidate}")
                        sm = tf.saved_model.load(candidate)
                        sigs = sm.signatures
                        if not sigs:
                            raise ValueError("SavedModel에 signatures가 없습니다.")
                        call_ep = os.getenv('TF_SERVING_ENDPOINT', 'serving_default')
                        if call_ep not in sigs:
                            call_ep = list(sigs.keys())[0]
                            print(f"요청한 endpoint가 없어 첫 번째 시그니처로 사용: {call_ep}")
                        func = sigs[call_ep]
                        _, input_kwargs = func.structured_input_signature
                        in_name, in_spec = list(input_kwargs.items())[0]
                        if in_spec.shape.rank == 4 and in_spec.shape[1] is not None:
                            expected = int(in_spec.shape[1])
                            if expected != self.img_size:
                                print(f"입력 크기 {self.img_size}→{expected}로 동기화")
                                self.img_size = expected
                        self._saved_signature = func
                        self._saved_input_name = in_name
                        self.model = None
                        print(f"SavedModel(signature) 로드 완료(폴백): {candidate}, endpoint={call_ep}, input={in_name}")
                    else:
                        raise e_h5
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            raise

    def _load_class_names(self):
        """환경변수(CLASS_NAMES)에서 클래스 이름 로드"""
        raw = os.getenv("CLASS_NAMES")
        if not raw:
            print("CLASS_NAMES 환경변수가 설정되지 않았습니다 (.env 확인).")
            self.class_names = ["happy", "sad"]
            print(f"기본 클래스 이름 사용: {self.class_names}")
            return

        names = [name.strip() for name in raw.split(',') if name.strip()]
        if not names:
            print("CLASS_NAMES 파싱 결과가 비어 있습니다. 예: CLASS_NAMES=happy,sad")
            self.class_names = ["happy", "sad"]
            print(f"기본 클래스 이름 사용: {self.class_names}")
            return

        # 중복 제거(순서 유지)
        seen = set()
        deduped = []
        for n in names:
            if n not in seen:
                seen.add(n)
                deduped.append(n)

        self.class_names = deduped
        print(f"클래스 이름 로드 완료(환경변수): {self.class_names}")

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
                print("특징 추출기 모드: 분류 결과 없음")
                return "Feature", 1.0, predictions[0]

            # 분류 모드인 경우
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            predicted_class = self.class_names[predicted_class_idx]

            return predicted_class, confidence, predictions[0]

        except Exception as e:
            print(f"감정 예측 실패: {e}")
            return "Error", 0.0, []

app = FastAPI()

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8001",
        "http://127.0.0.1:8001",
        "http://0.0.0.0:8001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# Initialize EmotionTester
tester = None

def get_tester():
    global tester
    if tester is None:
        if os.path.exists(MODEL_PATH):
            tester = EmotionTester(MODEL_PATH)
            print(f"Model loaded from {MODEL_PATH}")
        else:
            print(f"Model not found at {MODEL_PATH}")
    return tester

@app.get("/")
async def read_index():
    return FileResponse("src/static/index.html")

@app.websocket("/ws/debug")
async def websocket_debug_endpoint(websocket: WebSocket):
    print("Debug WebSocket connection attempt...")
    await websocket.accept()
    print("Debug WebSocket accepted.")
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Debug received: {data}")
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        print("Debug Client disconnected")
    except Exception as e:
        print(f"Debug Error: {e}")

@app.websocket("/ws/emotion")
async def websocket_endpoint(websocket: WebSocket):
    print("WebSocket connection attempt...")
    await websocket.accept()
    print("WebSocket accepted.")
    
    print(f"Loading model from: {MODEL_PATH}")
    current_tester = get_tester()
    if current_tester is None:
        print("Model not loaded. Closing connection.")
        await websocket.close(code=1011, reason="Model not loaded")
        return
    print("Model loaded successfully.")

    try:
        while True:
            # Receive data
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if 'image' not in message:
                continue
                
            # Decode image
            img_bytes = base64.b64decode(message['image'])
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue

            # Detect faces
            # EmotionTester.detect_faces expects BGR image (which cv2.imdecode returns)
            faces = current_tester.detect_faces(frame)
            
            print(f"Frame received. Faces detected: {len(faces)}")

            results = []
            for (x, y, w, h) in faces:
                # Extract face ROI
                face_img = frame[y:y+h, x:x+w]
                
                # Predict emotion
                emotion, confidence, probs = current_tester.predict_emotion(face_img)
                print(f"Emotion: {emotion}, Confidence: {confidence}")
                
                # Create probability dict
                prob_dict = {
                    name: float(prob) 
                    for name, prob in zip(current_tester.class_names, probs)
                }

                results.append({
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "emotion": emotion,
                    "confidence": float(confidence),
                    "probabilities": prob_dict
                })
            
            # Send results
            await websocket.send_json({"results": results})
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    uvicorn.run("src.server:app", host=SERVER_HOST, port=SERVER_PORT, reload=False)
