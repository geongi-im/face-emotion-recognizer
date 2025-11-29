import logging
import os
from datetime import datetime


def get_logger(name: str = __name__, log_level: int = logging.INFO) -> logging.Logger:
    """
    단일 로거 생성 함수 (파일명은 자동으로 로그 메시지에 포함)

    Args:
        name: 로거 이름 (보통 __name__ 사용 - 파일 경로가 자동으로 포함됨)
        log_level: 로그 레벨 (기본값: INFO)

    Returns:
        설정된 로거 객체
    """
    # 현재 날짜로 로그 파일명 생성
    today = datetime.now().strftime("%Y_%m_%d")
    log_filename = f"app_{today}.log"

    # 로그 파일 전체 경로 설정 (프로젝트 루트/logs/)
    project_root = os.path.dirname(os.path.dirname(__file__))
    logs_dir = os.path.join(project_root, "logs")
    log_filepath = os.path.join(logs_dir, log_filename)

    # logs 폴더가 없으면 생성
    os.makedirs(logs_dir, exist_ok=True)

    # 기존 핸들러가 있으면 제거 (중복 방지)
    logger = logging.getLogger(name)
    if logger.handlers:
        logger.handlers.clear()

    # 로거 레벨 설정
    logger.setLevel(log_level)

    # 포맷터 설정
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 파일 핸들러 설정
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 부모 로거로 전파 방지 (중복 로그 방지)
    logger.propagate = False

    logger.info(f"로거 설정 완료 - 로그 파일: {log_filepath}")

    return logger
