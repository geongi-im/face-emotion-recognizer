import json
import os
import time
import uuid
from pathlib import Path
from statistics import median
from typing import List, Tuple, Optional, Dict, Any
import cv2
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
from logger_config import get_logger

# 경로 설정 - 상위에서 경로 지정
JSON_FILE_PATH = r"..source\[라벨]EMOIMG_기쁨_TRAIN\img_emotion_training_data(기쁨).json"  # JSON 파일 경로
IMAGE_FOLDER_PATH = r"..source\[원천]EMOIMG_기쁨_TRAIN_01"  # 이미지 폴더 경로
OUTPUT_FOLDER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "train", "happy")  # 출력 폴더 경로
CSV_OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "img_emotion_training_data(기쁨).csv")  # CSV 출력 경로

# 로깅 설정
logger = get_logger(__name__)

def getMedianBox(boxes: List[Optional[Tuple[float, float, float, float]]]) -> Tuple[float, float, float, float]:
    """
    여러 어노테이션 박스에서 중앙값 박스 계산
    
    Args:
        boxes: (x1,y1,x2,y2) 튜플 리스트, None 값 포함 가능
        
    Returns:
        대표 박스 좌표 (float 튜플)
        - 1개 박스: 그대로 반환
        - 2개 박스: 각 좌표의 평균값 (중앙값)
        - 3개 이상: 각 좌표의 중앙값
    """
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        raise ValueError("no boxes provided")
    
    if len(boxes) == 1:
        x1, y1, x2, y2 = boxes[0]
        return (float(x1), float(y1), float(x2), float(y2))

    x1 = median(b[0] for b in boxes)
    y1 = median(b[1] for b in boxes)
    x2 = median(b[2] for b in boxes)
    y2 = median(b[3] for b in boxes)
    
    # 안전 체크: 좌표가 뒤바뀐 경우 정렬
    if x2 < x1: 
        x1, x2 = x2, x1
    if y2 < y1: 
        y1, y2 = y2, y1
        
    return (float(x1), float(y1), float(x2), float(y2))


def getSquarifyBox(box: Tuple[float, float, float, float], img_w: int, img_h: int, margin: float = 0.15) -> Tuple[float, float, float, float]:
    """
    사각형 박스를 정사각형으로 변환하고 여백 추가 후 이미지 경계에 클램프
    
    Args:
        box: 입력 바운딩 박스 (x1, y1, x2, y2)
        img_w: 이미지 너비
        img_h: 이미지 높이
        margin: 추가할 여백 비율 (기본값 0.15 = 15%)
        
    Returns:
        이미지 경계에 클램프된 정사각형 박스 좌표
    """
    x1, y1, x2, y2 = map(float, box)
    w, h = x2 - x1, y2 - y1
    side = max(w, h) * (1 + 2 * margin)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = side / 2.0
    
    # 중심점을 경계 내에 맞게 조정
    cx = min(max(cx, half), img_w - half)
    cy = min(max(cy, half), img_h - half)
    
    sx1, sy1 = cx - half, cy - half
    sx2, sy2 = cx + half, cy + half
    
    # 이미지 경계에 최종 클램프
    sx1 = max(0.0, sx1)
    sy1 = max(0.0, sy1)
    sx2 = min(float(img_w), sx2)
    sy2 = min(float(img_h), sy2)
    
    return (sx1, sy1, sx2, sy2)


def initialize_csv(csv_path: str) -> None:
    """
    CSV 파일 초기화 (헤더만 포함된 빈 파일 생성)
    
    Args:
        csv_path: CSV 파일 경로
    """
    df = pd.DataFrame(columns=[
        'new_filename', 'original_filename', 'gender', 'age', 
        'emotion', 'location', 'timestamp', 'status', 'error_reason'
    ])
    df.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"Initialized CSV file: {csv_path}")


def append_to_csv(csv_path: str, record: Dict[str, str]) -> None:
    """
    개별 처리 결과를 CSV 파일에 즉시 추가
    
    Args:
        csv_path: CSV 파일 경로
        record: 추가할 레코드 딕셔너리
    """
    try:
        df = pd.DataFrame([record])
        df.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8')
    except Exception as e:
        logger.error(f"Failed to append to CSV {csv_path}: {str(e)}")


def parse_filename(filename: str) -> Dict[str, str]:
    """
    원본 파일명에서 메타데이터 추출
    
    예시: "5f656a0f627a3ef96dec882437e3e7ada1c7a877201cf54dcd7a2c4508588ff3_여_30_기쁨_공공시설&종교&의료시설_20201204105732-001-007.jpg"
    
    Returns:
        hash, gender, age, emotion, location, timestamp 포함 딕셔너리
    """
    parts = filename.split('_')
    if len(parts) < 6:
        logger.warning(f"Unexpected filename format: {filename}")
        return {
            'hash': parts[0] if len(parts) > 0 else '',
            'gender': '',
            'age': '',
            'emotion': '',
            'location': '',
            'timestamp': ''
        }
    
    return {
        'hash': parts[0],
        'gender': parts[1],
        'age': parts[2],
        'emotion': parts[3],
        'location': parts[4],
        'timestamp': parts[5].split('.')[0]  # .jpg 확장자만 제거, 전체 업로드 번호 유지
    }


def crop_and_save_image(image_path: str, box: Tuple[float, float, float, float], output_path: str) -> bool:
    """
    이미지를 크롭하고 출력 경로에 저장
    
    Args:
        image_path: 원본 이미지 경로
        box: 바운딩 박스 좌표 (x1, y1, x2, y2)
        output_path: 크롭된 이미지 저장 경로
        
    Returns:
        성공 시 True, 실패 시 False
    """
    try:
        # OpenCV로 이미지 읽기 (유니코드 경로 처리)
        img_array = np.fromfile(image_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return False
        
        h, w = img.shape[:2]
        x1, y1, x2, y2 = box
        
        # 정수로 변환
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 이미지 크롭
        cropped = img[y1:y2, x1:x2]
        
        # 크롭된 이미지 저장 (유니코드 경로 처리)
        try:
            # 확장자 추출
            ext = os.path.splitext(output_path)[1]
            # 이미지 인코딩 후 파일 저장
            is_success, img_encoded = cv2.imencode(ext, cropped)
            if is_success:
                with open(output_path, 'wb') as f:
                    f.write(img_encoded.tobytes())
            else:
                logger.error(f"Failed to encode image: {output_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to save cropped image {output_path}: {str(e)}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error cropping image {image_path}: {str(e)}")
        return False


def process_emotion_data(json_path: str, image_folder: str, output_folder: str, csv_output: str) -> None:
    """
    메인 전처리 함수
    
    Args:
        json_path: JSON 어노테이션 파일 경로
        image_folder: 원본 이미지가 들어있는 폴더 경로
        output_folder: 크롭된 이미지 저장 폴더 경로
        csv_output: CSV 매핑 파일 저장 경로
    """
    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)
    
    # CSV 파일 초기화 (헤더 생성)
    initialize_csv(csv_output)
    
    # JSON 파일 로드
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded JSON file: {json_path}")
    except Exception as e:
        logger.error(f"Failed to load JSON file {json_path}: {str(e)}")
        return
    
    # 처리 완료 카운터
    processed_count = 0
    
    # 데이터의 각 항목 처리  
    total_items = len(data)
    for idx, item in enumerate(tqdm(data, desc="Processing images", unit="files"), 1):
        try:
            # 이미지 파일명 추출
            if 'filename' not in item:
                logger.warning(f"No filename in item: {item}")
                continue
                
            filename = item['filename']
            image_path = os.path.join(image_folder, filename)
            
            # 이미지 파일 존재 확인
            if not os.path.exists(image_path):
                logger.error(f"[{idx}/{total_items}] 이미지 파일을 찾을 수 없음: {image_path}")
                # 파일이 없어도 원본 정보는 CSV에 기록 (실패 상태로)
                metadata = parse_filename(filename)
                record = {
                    'new_filename': 'FAILED_FILE_NOT_FOUND',
                    'original_filename': filename,
                    'gender': metadata['gender'],
                    'age': metadata['age'],
                    'emotion': metadata['emotion'],
                    'location': metadata['location'],
                    'timestamp': metadata['timestamp'],
                    'status': 'FAILED',
                    'error_reason': 'Image file not found'
                }
                append_to_csv(csv_output, record)
                processed_count += 1
                continue
            
            # 어노테이션 박스 추출
            boxes = []
            for annot_key in ['annot_A', 'annot_B', 'annot_C']:
                if annot_key in item and item[annot_key]:
                    annot = item[annot_key]
                    # boxes 안에 있는 좌표 정보 확인
                    if 'boxes' in annot:
                        box_data = annot['boxes']
                        # minX, minY, maxX, maxY 형식으로 된 좌표를 x1, y1, x2, y2로 변환
                        if all(key in box_data for key in ['minX', 'minY', 'maxX', 'maxY']):
                            boxes.append((
                                float(box_data['minX']),  # x1
                                float(box_data['minY']),  # y1
                                float(box_data['maxX']),  # x2
                                float(box_data['maxY'])   # y2
                            ))
            
            if not boxes:
                logger.warning(f"[{idx}/{total_items}] 유효한 어노테이션 박스가 없음: {filename}")
                # 어노테이션 박스가 없어도 원본 정보는 CSV에 기록 (실패 상태로)
                metadata = parse_filename(filename)
                record = {
                    'new_filename': 'FAILED_NO_ANNOTATION',
                    'original_filename': filename,
                    'gender': metadata['gender'],
                    'age': metadata['age'],
                    'emotion': metadata['emotion'],
                    'location': metadata['location'],
                    'timestamp': metadata['timestamp'],
                    'status': 'FAILED',
                    'error_reason': 'No valid annotation boxes'
                }
                append_to_csv(csv_output, record)
                processed_count += 1
                continue
            
            # 중앙값 박스 계산
            median_box = getMedianBox(boxes)
            
            # 이미지 크기 가져오기 (유니코드 경로 처리)
            try:
                img_array = np.fromfile(image_path, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            except Exception as e:
                logger.error(f"Exception reading image file {image_path}: {str(e)}")
                img = None
                
            if img is None:
                logger.error(f"[{idx}/{total_items}] 이미지 크기 확인을 위한 로드 실패: {image_path}")
                # 이미지 로드 실패해도 원본 정보는 CSV에 기록 (실패 상태로)
                metadata = parse_filename(filename)
                record = {
                    'new_filename': 'FAILED_IMAGE_LOAD',
                    'original_filename': filename,
                    'gender': metadata['gender'],
                    'age': metadata['age'],
                    'emotion': metadata['emotion'],
                    'location': metadata['location'],
                    'timestamp': metadata['timestamp'],
                    'status': 'FAILED',
                    'error_reason': 'Failed to load image'
                }
                append_to_csv(csv_output, record)
                processed_count += 1
                continue
            
            img_h, img_w = img.shape[:2]
            
            # 정사각형 박스로 변환
            square_box = getSquarifyBox(median_box, img_w, img_h)
            
            # 새 파일명 생성 (해시 + 업로드 번호)
            metadata = parse_filename(filename)
            # 업로드 번호 추출 (timestamp 부분에서 추출)
            upload_number = metadata['timestamp']  # 이미 parse_filename에서 추출됨
            new_filename = f"crop_{metadata['hash']}_{upload_number}.jpg"
            output_path = os.path.join(output_folder, new_filename)
            
            # 기존 크롭 파일이 있는지 확인 (정확히 같은 파일명)
            if os.path.exists(output_path):
                logger.info(f"[{idx}/{total_items}] 건너뜀 - 이미 처리된 파일: {filename} -> {new_filename}")
                # CSV에 기존 파일 정보 기록
                record = {
                    'new_filename': new_filename,
                    'original_filename': filename,
                    'gender': metadata['gender'],
                    'age': metadata['age'],
                    'emotion': metadata['emotion'],
                    'location': metadata['location'],
                    'timestamp': metadata['timestamp'],
                    'status': 'SKIPPED',
                    'error_reason': 'Already processed'
                }
                append_to_csv(csv_output, record)
                processed_count += 1
                continue
            
            # 이미지 크롭 및 저장
            if crop_and_save_image(image_path, square_box, output_path):
                # CSV에 즉시 추가 (성공)
                record = {
                    'new_filename': new_filename,
                    'original_filename': filename,
                    'gender': metadata['gender'],
                    'age': metadata['age'],
                    'emotion': metadata['emotion'],
                    'location': metadata['location'],
                    'timestamp': metadata['timestamp'],
                    'status': 'SUCCESS',
                    'error_reason': ''
                }
                append_to_csv(csv_output, record)
                logger.info(f"[{idx}/{total_items}] 성공적으로 처리됨: {filename} -> {new_filename}")
            else:
                # 크롭 실패해도 원본 정보는 CSV에 기록 (실패 상태로)
                record = {
                    'new_filename': 'FAILED_CROP_ERROR',
                    'original_filename': filename,
                    'gender': metadata['gender'],
                    'age': metadata['age'],
                    'emotion': metadata['emotion'],
                    'location': metadata['location'],
                    'timestamp': metadata['timestamp'],
                    'status': 'FAILED',
                    'error_reason': 'Failed to crop and save image'
                }
                append_to_csv(csv_output, record)
                logger.error(f"[{idx}/{total_items}] 처리 실패: {filename}")
            
            processed_count += 1
                
        except Exception as e:
            logger.error(f"Error processing item {item}: {str(e)}")
            # 예외 발생해도 원본 정보는 CSV에 기록 (실패 상태로)
            try:
                filename = item.get('filename', 'UNKNOWN_FILENAME')
                metadata = parse_filename(filename)
                record = {
                    'new_filename': 'FAILED_EXCEPTION',
                    'original_filename': filename,
                    'gender': metadata['gender'],
                    'age': metadata['age'],
                    'emotion': metadata['emotion'],
                    'location': metadata['location'],
                    'timestamp': metadata['timestamp'],
                    'status': 'FAILED',
                    'error_reason': f'Exception: {str(e)}'
                }
                append_to_csv(csv_output, record)
                processed_count += 1
            except Exception as nested_e:
                logger.error(f"Failed to record error info: {str(nested_e)}")
            continue
    
    # 처리 완료 통계
    logger.info(f"=== 처리 완료 ===")
    logger.info(f"CSV 파일 저장됨: {csv_output}")
    logger.info(f"총 처리된 이미지: {processed_count}/{total_items}")
    logger.info(f"처리 완료율: {processed_count/total_items*100:.1f}%")


if __name__ == "__main__":
    # 경로 존재 확인
    if not os.path.exists(JSON_FILE_PATH):
        logger.error(f"JSON file not found: {JSON_FILE_PATH}")
        exit(1)
    
    if not os.path.exists(IMAGE_FOLDER_PATH):
        logger.error(f"Image folder not found: {IMAGE_FOLDER_PATH}")
        exit(1)
    
    # 전처리 실행
    logger.info("Starting image preprocessing...")
    process_emotion_data(
        json_path=JSON_FILE_PATH,
        image_folder=IMAGE_FOLDER_PATH,
        output_folder=OUTPUT_FOLDER_PATH,
        csv_output=CSV_OUTPUT_PATH
    )
    logger.info("Image preprocessing completed!")