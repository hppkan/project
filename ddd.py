import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
from collections import Counter
import subprocess
from plus import group_objects


# --- 파일 선택창 열기 ---
Tk().withdraw()
image_path = askopenfilename(
    title="사진 파일을 선택하세요",
    filetypes=[
        ("JPEG files", "*.jpg"),
        ("JPEG files", "*.jpeg"),
        ("PNG files", "*.png"),
        ("All files", "*.*")
    ]
)

if not image_path:
    print("이미지를 선택하지 않았습니다.")
    exit()

print("선택된 이미지:", image_path)

# read image
img = cv2.imread(image_path)
if img is None:
    print("이미지를 불러오지 못했습니다. 경로를 확인하세요.")
    exit()

height, width, channel = img.shape
print('original image shape:', height, width, channel)

# get blob from image
blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
print('blob shape:', blob.shape)

# read coco object names
# Downloads 폴더 경로 설정
downloads_path = os.path.join(os.path.expanduser("~"), "Downloads", "coco.names")
current_path = "coco.names"

# Downloads 폴더 또는 현재 폴더에서 파일 찾기
if os.path.exists(current_path):
    coco_names_path = current_path
elif os.path.exists(downloads_path):
    coco_names_path = downloads_path
    print(f"Downloads 폴더에서 coco.names 파일을 찾았습니다: {downloads_path}")
else:
    print("coco.names 파일을 찾을 수 없습니다. 자동으로 생성합니다...")
    classes = [
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    with open(current_path, "w") as f:
        f.write("\n".join(classes))
    coco_names_path = current_path

with open(coco_names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]
print('number of classes =', len(classes))

# YOLO 파일 찾기
def find_file(filename):
    current_path = filename
    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads", filename)
    
    if os.path.exists(current_path):
        return current_path
    elif os.path.exists(downloads_path):
        print(f"Downloads 폴더에서 {filename} 파일을 찾았습니다")
        return downloads_path
    else:
        raise FileNotFoundError(f"{filename} 파일을 찾을 수 없습니다")

cfg_path = find_file("yolov3.cfg")
weights_path = find_file("yolov3.weights")

# load pre-trained yolo model
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# set output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
print('output layers:', output_layers)

# detect objects
net.setInput(blob)
outs = net.forward(output_layers)

# get bounding boxes and confidence scores
class_ids = []
confidence_scores = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidence_scores.append(float(confidence))
            class_ids.append(class_id)

print('number of detected objects =', len(boxes))

# non maximum suppression
indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, 0.5, 0.4)
print('number of final objects =', len(indices))

# draw boxes and collect detected objects with positions
colors = np.random.uniform(0, 255, size=(len(classes), 3))
font = cv2.FONT_HERSHEY_PLAIN
detected_objects = []
object_details = []  # 위치 정보 포함

for i in range(len(boxes)):
    if i in indices:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        detected_objects.append(label)
        
        # 위치 정보 저장
        center_x = x + w // 2
        center_y = y + h // 2
        object_details.append({
            'label': label,
            'x': center_x,
            'y': center_y,
            'width': w,
            'height': h,
            'confidence': confidence_scores[i]
        })
        
        print(f'class {label} detected at {x}, {y}, {w}, {h}')
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10), font, 1, color, 2)

# 음성 설명 생성 (상세 버전)
def create_detailed_description(objects, details, img_width, img_height):
    if not objects:
        return "이미지에서 객체를 찾지 못했습니다."
    
    # 객체 개수 세기
    object_count = Counter(objects)
    
    # 한국어 객체명 매핑
    korean_names = {
        "person": "사람", "bicycle": "자전거", "car": "자동차", "motorbike": "오토바이",
        "aeroplane": "비행기", "bus": "버스", "train": "기차", "truck": "트럭", "boat": "보트",
        "traffic light": "신호등", "fire hydrant": "소화전", "stop sign": "정지 표지판",
        "parking meter": "주차 미터기", "bench": "벤치", "bird": "새", "cat": "고양이",
        "dog": "개", "horse": "말", "sheep": "양", "cow": "소", "elephant": "코끼리",
        "bear": "곰", "zebra": "얼룩말", "giraffe": "기린", "backpack": "백팩",
        "umbrella": "우산", "handbag": "핸드백", "tie": "넥타이", "suitcase": "여행가방",
        "frisbee": "프리스비", "skis": "스키", "snowboard": "스노보드", "sports ball": "공",
        "kite": "연", "baseball bat": "야구 방망이", "baseball glove": "야구 글러브",
        "skateboard": "스케이트보드", "surfboard": "서핑보드", "tennis racket": "테니스 라켓",
        "bottle": "병", "wine glass": "와인잔", "cup": "컵", "fork": "포크",
        "knife": "나이프", "spoon": "숟가락", "bowl": "그릇", "banana": "바나나",
        "apple": "사과", "sandwich": "샌드위치", "orange": "오렌지", "broccoli": "브로콜리",
        "carrot": "당근", "hot dog": "핫도그", "pizza": "피자", "donut": "도넛",
        "cake": "케이크", "chair": "의자", "sofa": "소파", "pottedplant": "화분",
        "bed": "침대", "diningtable": "식탁", "toilet": "변기", "tvmonitor": "모니터",
        "laptop": "노트북", "mouse": "마우스", "remote": "리모컨", "keyboard": "키보드",
        "cell phone": "휴대폰", "microwave": "전자레인지", "oven": "오븐", "toaster": "토스터",
        "sink": "싱크대", "refrigerator": "냉장고", "book": "책", "clock": "시계",
        "vase": "꽃병", "scissors": "가위", "teddy bear": "곰 인형", "hair drier": "헤어 드라이어",
        "toothbrush": "칫솔"
    }
    
    # 위치 분석 함수
    def get_position(x, y, w, h):
        pos_h = "중앙"
        pos_v = "중간"
        
        if x < img_width * 0.33:
            pos_h = "왼쪽"
        elif x > img_width * 0.66:
            pos_h = "오른쪽"
            
        if y < img_height * 0.33:
            pos_v = "위쪽"
        elif y > img_height * 0.66:
            pos_v = "아래쪽"
        
        return f"{pos_v} {pos_h}" if pos_v != "중간" or pos_h != "중앙" else "중앙"
    
    # 크기 분석
    def get_size(w, h):
        area = w * h
        img_area = img_width * img_height
        ratio = area / img_area
        
        if ratio > 0.3:
            return "큰"
        elif ratio > 0.1:
            return "중간 크기의"
        else:
            return "작은"

    # 객체 거리감 분석    
    def get_distance_level(w, h, img_w, img_h):
        area = w * h
        img_area = img_w * img_h
        ratio = area / img_area

        if ratio > 0.25:
            return "아주 가까운"
        elif ratio > 0.1:
            return "가까운"
        elif ratio > 0.04:
            return "보통 거리의"
        else:
            return "먼 거리의"
    
    # 기본 설명
    total = len(objects)
    description = f"이 이미지에는 총 {total}개의 객체가 있습니다. "
    
    # 주요 객체 설명
    main_objects = []
    for obj, count in object_count.most_common(3):  # 상위 3개
        korean_name = korean_names.get(obj, obj)
        if count > 1:
            main_objects.append(f"{korean_name} {count}개")
        else:
            main_objects.append(f"{korean_name}")
    
    description += "주요 객체는 " + ", ".join(main_objects) + "입니다. "
    
    # 상황 분석
    situation = analyze_situation(object_count, korean_names)
    if situation:
        description += situation + " "
    
    # 위치별 상세 설명 (최대 5개 객체)
    if len(details) <= 5:
        description += "위치를 자세히 보면, "
        position_desc = []
        for detail in details:
            korean_name = korean_names.get(detail['label'], detail['label'])
            position = get_position(detail['x'], detail['y'], detail['width'], detail['height'])
            size = get_size(detail['width'], detail['height'])
            distance = get_distance_level(detail['width'], detail['height'], img_width, img_height)
            position_desc.append(f"{position}에 {distance} {size} {korean_name}")
        description += ", ".join(position_desc) + "이 있습니다."
    
    return description

def analyze_situation(object_count, korean_names):
    """객체 조합으로 상황 추론"""
    objects_set = set(object_count.keys())
    
    # 교통 관련
    if {'car', 'truck', 'bus'} & objects_set:
        return "도로나 주차장 같은 교통 환경으로 보입니다."
    
    # 식사 관련
    if {'fork', 'knife', 'spoon', 'bowl', 'cup'} & objects_set or \
       {'pizza', 'sandwich', 'cake', 'donut'} & objects_set:
        return "식사나 음식과 관련된 장면입니다."
    
    # 실내 거실
    if {'sofa', 'chair', 'tvmonitor'} & objects_set:
        return "거실이나 실내 공간으로 보입니다."
    
    # 침실
    if {'bed'} & objects_set:
        return "침실 환경입니다."
    
    # 사무실
    if {'laptop', 'keyboard', 'mouse'} & objects_set:
        return "사무실이나 작업 공간으로 보입니다."
    
    # 야외 활동
    if {'frisbee', 'sports ball', 'kite', 'surfboard', 'skateboard'} & objects_set:
        return "야외 활동이나 레저 장면입니다."
    
    # 동물
    if {'dog', 'cat', 'bird', 'horse'} & objects_set:
        animals = [korean_names.get(obj, obj) for obj in objects_set if obj in {'dog', 'cat', 'bird', 'horse'}]
        return f"{', '.join(animals)}가 있는 장면입니다."
    
    # 사람이 많은 경우
    if 'person' in object_count and object_count['person'] >= 3:
        return "여러 사람이 모여 있는 장면입니다."
    
    return None

# 설명 생성
description = create_detailed_description(detected_objects, object_details, width, height)
print("\n=== 음성 설명 ===")
print(description)

# macOS의 'say' 명령어로 음성 출력
print("음성을 재생합니다...")
try:
    # macOS 기본 TTS 사용 (한국어 음성: Yuna)
    subprocess.run(['say', '-v', 'Yuna', description], check=True)
except subprocess.CalledProcessError:
    print("음성 재생에 실패했습니다. 기본 음성을 사용합니다.")
    subprocess.run(['say', description])
except FileNotFoundError:
    print("say 명령어를 찾을 수 없습니다. macOS가 아닌 것 같습니다.")
    # pyttsx3 fallback
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(description)
        engine.runAndWait()
    except:
        print("음성 재생을 건너뜁니다.")

# 이미지 표시
cv2.imshow('Objects', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
