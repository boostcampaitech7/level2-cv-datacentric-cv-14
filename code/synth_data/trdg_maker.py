from trdg.generators import GeneratorFromStrings
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from synth_utils import make_document, Document
import os
import time
import json

def get_words(count=128):
    words = []
    word_dict = []
    with open('./lang_example/japanese_receipt_dict.txt', 'r', encoding='utf-8') as f:
        for line in f:
            words_arr = line.strip().split()
            word_dict.extend(words_arr)

    np.random.shuffle(word_dict)
    
    for idx in range(count):
        margins = [np.random.randint(3, 8) for _ in range(4)]  # 상하좌우 각각 다른 마진 생성
        character_spacing = np.random.randint(1, 30)
        size = np.random.randint(64, 200)  # 이미지 크기를 64~200 사이에서 랜덤하게 설정
        generator = GeneratorFromStrings(
            strings=[word_dict[idx]],
            language="ja", 
            count=1,
            background_type=0,
            fit=True,
            margins=tuple(margins),  # 4개의 다른 마진 값 전달
            size=size,  # 랜덤한 이미지 크기 적용
            character_spacing=character_spacing,  # 랜덤한 글자 간격 적용
            blur=3,
            random_blur=True,
            distorsion_type=1,
        )
        for _, (patch, text) in enumerate(generator):
            words.append({"patch": patch, "text": text, "size": patch.size, "margins": margins})
    return words

def vm_function(bbox: list[float], M: np.ndarray) -> np.ndarray:
    v = np.array(bbox).reshape(-1, 2).T
    v = np.vstack([v, np.ones((1, 4))])
    v = np.dot(M, v)
    v = v[:2] / v[2]
    out = v.T.flatten().tolist()
    return out

def perturb_document_inplace(document: Document, pad=0, color=None) -> Document:
    if color is None:
        color = [64, 64, 64]
    width, height = np.array(document["image"].size)
    magnitude_lb = 0
    magnitude_ub = 200
    src = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    perturb = np.random.uniform(magnitude_lb, magnitude_ub, (4, 2)) * np.array(
        [[1, 1], [-1, 1], [-1, -1], [1, -1]]
    )
    perturb = perturb.astype(np.float32)
    dst = src + perturb

    # obtain the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # transform the image
    out = cv2.warpPerspective(
        np.array(document["image"]),
        M,
        document["image"].size,
        flags=cv2.INTER_LINEAR,
        borderValue=color,
    )
    out = Image.fromarray(out)
    document["image"] = out

    # transform the bounding boxes
    for word in document["words"]:
        bbox = word["bbox"]

        word["bbox"] = vm_function(bbox, M)

    return document


# 디렉토리 생성
os.makedirs("./synth_receipt_data/japanese/img/train", exist_ok=True)
os.makedirs("./synth_receipt_data/japanese/train/ufo", exist_ok=True)

# UFO format json 데이터 초기화
ufo_data = {"images": {}}

# 이미지 개수가 100개가 될 때까지 생성
while len(os.listdir("./synth_receipt_data/japanese/img/train")) < 100:
    # 문서 생성 및 변형
    doc = make_document(get_words(30))
    perturb_document_inplace(doc)
    
    # 이미지 파일명 생성
    img_filename = f"synthetic_image_{time.strftime('%Y%m%d_%H%M%S')}_{len(os.listdir('./synth_receipt_data/japanese/img/train')):03d}.jpg"
    img_path = f"./synth_receipt_data/japanese/img/train/{img_filename}"
    
    # 이미지 저장
    doc["image"].save(img_path)
    
    # 현재 이미지에 대한 UFO 데이터 생성
    ufo_data["images"][img_filename] = {
        "paragraphs": {},
        "words": {},
        "chars": {},
        "img_w": doc["image"].size[0],
        "img_h": doc["image"].size[1],
        "num_patches": None,
        "tags": [],
        "relations": {},
        "annotation_log": {
            "worker": "worker",
            "timestamp": time.strftime("%Y-%m-%d"),
            "tool_version": "",
            "source": None
        },
        "license_tag": {
            "usability": True,
            "public": False,
            "commercial": True,
            "type": None,
            "holder": "kgs"
        }
    }
    
    # words 정보 추가
    for idx, word in enumerate(doc["words"], 1):
        word_id = str(idx).zfill(4)
        points = [[word["bbox"][i], word["bbox"][i+1]] for i in range(0, 8, 2)]
        
        ufo_data["images"][img_filename]["words"][word_id] = {
            "transcription": word["text"],
            "points": points
        }
    
    # 처리 시간을 위한 딜레이
    time.sleep(1)

# 모든 이미지에 대한 JSON 파일 저장
json_path = "./synth_receipt_data/japanese/train/ufo/train.json"
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(ufo_data, f, ensure_ascii=False, indent=4)
