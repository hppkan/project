# --- YOLO 관련 설정 ---
YOLO_WEIGHTS_PATH = "yolov3.weights"
YOLO_CFG_PATH = "yolov3.cfg"
COCO_NAMES_PATH = "coco.names"

# 신뢰도 임계값
CONFIDENCE_THRESHOLD = 0.5 
NMS_THRESHOLD = 0.4 # 비최대 억제(겹치는 박스 제거) 임계값

# --- 크기 분석 임계값 ---
SIZE_LARGE_THRESHOLD = 0.30  # 이미지의 30% 이상
SIZE_SMALL_THRESHOLD = 0.10  # 이미지의 10% 미만

# --- TTS 설정 ---
MACOS_VOICE_NAME = "Yuna" # macOS 한국어 음성

# COCO 데이터셋 80개 클래스에 대한 영한 번역 사전
# (전체 80개를 다 채워야 하지만, 예시를 위해 일부만 작성했습니다.)
ENG_TO_KOR_MAP = {
    "person": "사람",
    "bicycle": "자전거",
    "car": "자동차",
    "motorbike": "오토바이",
    "aeroplane": "비행기",
    "bus": "버스",
    "train": "기차",
    "truck": "트럭",
    "boat": "보트",
    "traffic light": "신호등",
    "fire hydrant": "소화전",
    "stop sign": "정지 표지판",
    "bird": "새",
    "cat": "고양이",
    "dog": "개",
    "horse": "말",
    "sheep": "양",
    "cow": "소",
    "backpack": "가방",
    "umbrella": "우산",
    "handbag": "핸드백",
    "tie": "넥타이",
    "suitcase": "여행 가방",
    "fork": "포크",
    "knife": "나이프",
    "spoon": "숟가락",
    "bowl": "그릇",
    "chair": "의자",
    "sofa": "소파",
    "bed": "침대",
    "diningtable": "식탁",
    "toilet": "화장실",
    "tvmonitor": "TV",
    "laptop": "노트북",
    "mouse": "마우스",
    "remote": "리모컨",
    "keyboard": "키보드",
    "cell phone": "휴대전화"
}

def translate_label(eng_label):
    """영문 레이블을 받아 한글로 변환합니다. 없으면 원래 영문 반환."""
    return ENG_TO_KOR_MAP.get(eng_label, eng_label)

from collections import Counter

def group_objects(detected_labels):
    """
    탐지된 객체 리스트를 받아 같은 객체끼리 묶어 개수를 세고
    사용자에게 설명용 텍스트까지 생성하는 기능.

    예) ['person', 'car', 'car', 'dog']
    → {'person': 1, 'car': 2, 'dog': 1}
    → '사람 1개, 자동차 2개, 개 1개'
    """
    if not detected_labels:
        return {}, "탐지된 객체가 없습니다."

    counts = Counter(detected_labels)

    # 한글 변환 + 개수 설명
    description_parts = []
    for eng, count in counts.items():
        kor = ENG_TO_KOR_MAP.get(eng, eng)
        description_parts.append(f"{kor} {count}개")

    description_text = ", ".join(description_parts)

    return counts, description_text


# 상황 설명

def infer_scene_context(detected_labels):
    """
    탐지된 영문 객체 이름 리스트(set 권장)를 받아 상황을 추론합니다.
    detected_labels: 예) {'person', 'car', 'bus'}
    """
    labels_set = set(detected_labels) # 집합 연산을 위해 set으로 변환

    scene = "알 수 없는 상황"

    # --- 상황별 규칙 정의 ---
    
    # 🚗 교통: 자동차, 버스, 트럭 중 하나라도 있으면
    if labels_set.intersection({'car', 'bus', 'truck', 'traffic light'}):
        scene = "도로나 주차장"
        
    # 🛋️ 거실: 소파, TV, 의자 조합 체크 (소파나 TV가 핵심)
    elif labels_set.intersection({'sofa', 'tvmonitor'}) and labels_set.intersection({'chair', 'table'}):
         scene = "거실 공간"

    # 🛏️ 침실: 침대가 있으면 강력한 단서
    elif 'bed' in labels_set:
        scene = "침실 환경"

    # 🍕 식사: 식기류나 식탁이 있는 경우
    elif labels_set.intersection({'fork', 'knife', 'spoon', 'bowl', 'diningtable'}):
        scene = "식사 장면"

    # 💻 사무실: 업무 관련 기기
    elif labels_set.intersection({'laptop', 'keyboard', 'mouse'}):
        scene = "작업 공간"
    
    # 🐕 동물: 동물들이 있는 경우
    elif labels_set.intersection({'dog', 'cat', 'bird', 'horse', 'sheep', 'cow'}):
        scene = "동물이 있는 장면"
        
    # 🏃 야외: 야외 활동 관련 물품 (예시 외 추가)
    elif labels_set.intersection({'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'skateboard', 'surfboard'}):
        scene = "야외 활동"

    # 👥 모임: 사람 수 체크 (이건 labels_set 만으로는 안 되고, 별도 카운트가 필요하지만 일단 규칙에 포함)
    # (주의: 이 부분은 ddd.py에서 사람 수를 세어서 넘겨주는 방식이 더 정확합니다.)
    elif detected_labels.count('person') >= 3:
         scene = "여러 사람이 모인 장면"

    return scene

# OS별 음성 출력 분기
import platform
import os
from config import MACOS_VOICE_NAME

# 윈도우/리눅스용 pyttsx3 (설치되어 있을 때만 로드)
try:
    import pyttsx3
    tts_engine = pyttsx3.init()
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("경고: pyttsx3가 설치되지 않아 macOS 외 OS에서는 소리가 나지 않습니다.")


def speak(text):
    """OS를 감지하여 적절한 방식으로 텍스트를 음성으로 출력합니다."""
    current_os = platform.system()
    print(f"[🔊 TTS 출력] {text}") # 로그 출력

    if current_os == 'Darwin':  # macOS
        # say 명령어 사용, Yuna 목소리 지정, 백그라운드 실행(&)
        try:
            os.system(f'say -v {MACOS_VOICE_NAME} "{text}" &')
        except Exception as e:
            print(f"macOS TTS 오류: {e}")

    elif current_os == 'Windows' or current_os == 'Linux':
        if PYTTSX3_AVAILABLE:
            try:
                tts_engine.say(text)
                tts_engine.runAndWait()
            except Exception as e:
                 print(f"pyttsx3 TTS 오류: {e}")
        else:
            print("알림: TTS 라이브러리가 없어 음성 출력을 건너뜁니다.")
    
    else:
        print(f"알림: 지원하지 않는 OS({current_os})입니다.")

# 테스트 실행
if __name__ == '__main__':

    speak("안녕하세요. 음성 출력 테스트입니다.")
