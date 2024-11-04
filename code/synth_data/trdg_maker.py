from trdg.generators import GeneratorFromStrings
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from synth_utils import make_document, Document
import os
import time
import json
from typing import List, Dict

LANGUAGE_CONFIGS = {
    'chinese': {
        'dict_path': './lang_example/chinese_receipt_dict.txt',
        'lang_code': 'cn',
        'font_path': None
    },
    'japanese': {
        'dict_path': './lang_example/japanese_receipt_dict.txt', 
        'lang_code': 'ja',
        'font_path': None
    },
    'vietnamese': {
        'dict_path': './lang_example/vietnamese_receipt_dict.txt',
        'lang_code': 'vi',
        'font_path': './fonts/RobotoMono-Italic-VariableFont_wght.ttf'
    },
    'thai': {
        'dict_path': './lang_example/thai_receipt_dict.txt',
        'lang_code': 'th',
        'font_path': './fonts/Itim-Regular.ttf'
    }
}

def get_words(count: int, language: str) -> List[Dict]:
    """여러 언어의 단어 목록을 생성하고 랜덤하게 이미지로 변환"""
    words = []
    word_dict = []
    
    config = LANGUAGE_CONFIGS[language]
    
    # 단어 사전 파일 읽기
    with open(config['dict_path'], 'r', encoding='utf-8') as f:
        for line in f:
            word_dict.extend(line.strip().split())

    np.random.shuffle(word_dict)
    
    # 각 단어에 대해 이미지 생성
    for idx in range(count):
        margins = [np.random.randint(3, 8) for _ in range(4)]
        generator = GeneratorFromStrings(
            strings=[word_dict[idx]],
            language=config['lang_code'],
            count=1, 
            background_type=0,
            fit=True,
            margins=tuple(margins),
            size=np.random.randint(64, 200),
            character_spacing=np.random.randint(1, 30),
            blur=3,
            random_blur=True,
            distorsion_type=1,
            fonts=[config['font_path']] if config['font_path'] else []
        )
        
        for _, (patch, text) in enumerate(generator):
            words.append({
                "patch": patch,
                "text": text,
                "size": patch.size,
                "margins": margins
            })
    return words

def vm_function(bbox: list[float], M: np.ndarray) -> np.ndarray:
    """바운딩 박스 좌표를 변환 행렬에 따라 변환"""
    v = np.array(bbox).reshape(-1, 2).T
    v = np.vstack([v, np.ones((1, 4))])
    v = np.dot(M, v)
    v = v[:2] / v[2]
    return v.T.flatten().tolist()

def perturb_document_inplace(document: Document, pad=0, color=None) -> Document:
    """문서 이미지에 원근 변환 적용"""
    color = color or [64, 64, 64]
    width, height = np.array(document["image"].size)
    
    # 원근 변환을 위한 소스/타겟 포인트 설정
    src = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    perturb = np.random.uniform(0, 200, (4, 2)) * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    dst = src + perturb.astype(np.float32)

    # 변환 행렬 계산 및 이미지 변환
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(
        np.array(document["image"]),
        M,
        document["image"].size,
        flags=cv2.INTER_LINEAR,
        borderValue=color
    )
    document["image"] = Image.fromarray(out)

    # 바운딩 박스 좌표 변환
    for word in document["words"]:
        word["bbox"] = vm_function(word["bbox"], M)

    return document

def create_ufo_entry(img_filename: str, doc: Document) -> dict:
    """UFO 포맷의 이미지 엔트리 생성"""
    entry = {
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
    
    # 단어 정보 추가
    for idx, word in enumerate(doc["words"], 1):
        word_id = str(idx).zfill(4)
        points = [[word["bbox"][i], word["bbox"][i+1]] for i in range(0, 8, 2)]
        entry["words"][word_id] = {
            "transcription": word["text"],
            "points": points
        }
        
    return entry

def generate_synthetic_data(images_per_language: int = 100):
    """각 언어별로 지정된 수만큼 합성 데이터를 생성하여 하나의 디렉토리에 저장"""
    # 디렉토리 생성
    os.makedirs("./synth_receipt/img/train", exist_ok=True)
    os.makedirs("./synth_receipt/train/ufo", exist_ok=True)

    # 합성 데이터 생성
    ufo_data = {"images": {}}
    current_count = 0
    
    # 각 언어별로 이미지 생성
    for language in LANGUAGE_CONFIGS.keys():
        print(f"Generating {images_per_language} images for {language}...")
        
        for i in range(images_per_language):
            # 문서 생성 및 변형
            doc = make_document(get_words(30, language))
            perturb_document_inplace(doc)
            
            # 이미지 저장
            img_filename = f"{language}.synthetic.{current_count:03d}.jpg"
            img_path = f"./synth_receipt/img/train/{img_filename}"
            doc["image"].save(img_path)
            
            # UFO 데이터 추가
            ufo_data["images"][img_filename] = create_ufo_entry(img_filename, doc)
            current_count += 1
            time.sleep(1)
            
        print(f"Completed generating {images_per_language} images for {language}")

    # UFO 데이터 저장
    json_path = "./synth_receipt/train/ufo/train.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(ufo_data, f, ensure_ascii=False, indent=4)

# 각 언어별로 100개씩 데이터 생성
print("Starting synthetic data generation...")
generate_synthetic_data()
print("Completed generating all data")
