import json
import os
import os.path as osp
import numpy as np
from PIL import Image

def convert_ufo_to_coco(ufo_dir, img_dir, output_path):
    """UFO 형식의 데이터를 COCO 형식으로 변환하는 함수
    
    Args:
        ufo_dir: UFO json 파일이 있는 디렉토리 경로
        img_dir: 이미지 파일이 있는 디렉토리 경로 
        output_path: 변환된 COCO json을 저장할 경로
    """
    
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "text"}]
    }
    
    annotation_id = 1
    
    # UFO json 파일 읽기
    with open(ufo_dir, 'r', encoding='utf-8') as f:
        ufo_data = json.load(f)
        
    # 이미지와 어노테이션 정보 변환
    for img_id, (img_name, img_info) in enumerate(ufo_data['images'].items(), 1):
        # 이미지 정보
        img_path = osp.join(img_dir, img_name)
        
        # UFO 데이터에서 이미지 크기 정보 가져오기
        width = img_info.get('img_w', None)
        height = img_info.get('img_h', None)
        
        # 이미지 크기 정보가 없는 경우 이미지 파일에서 직접 읽기
        if width is None or height is None:
            img = Image.open(img_path)
            width, height = img.size
        
        img_info_coco = {
            "id": img_id,
            "file_name": img_name,
            "width": width,
            "height": height,
            "date_captured": img_info.get('annotation_log', {}).get('timestamp', ""),
            "tags": img_info.get('tags', [])
        }
        coco_format["images"].append(img_info_coco)
        
        # 단어 단위 박스 정보 변환
        for word_id, word_info in img_info['words'].items():
            points = np.array(word_info['points'])
            
            # COCO 형식의 bbox 계산 [x, y, width, height]
            x1, y1 = points.min(axis=0)
            x2, y2 = points.max(axis=0)
            bbox = [float(x1), float(y1), float(x2-x1), float(y2-y1)]
            
            # Segmentation 포인트
            segmentation = [[float(p[0]), float(p[1])] for p in points]
            
            # 면적 계산
            area = float((x2-x1) * (y2-y1))
            
            annotation = {
                "id": annotation_id,
                "image_id": img_id,
                "category_id": 1,
                "bbox": bbox,
                "segmentation": segmentation,
                "area": area,
                "iscrowd": 0,
                "attributes": {
                    "transcription": word_info['transcription'],
                    "chars": word_info.get('chars', {}),
                    "relations": word_info.get('relations', {})
                }
            }
            
            coco_format["annotations"].append(annotation)
            annotation_id += 1
            
    # COCO json 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_format, f, indent=4, ensure_ascii=False)


# 모든 언어 및 split에 대해 변환 수행
LANGUAGE_LIST = ['chinese', 'japanese', 'thai', 'vietnamese']

for lang in LANGUAGE_LIST:
    ufo_path = f"/data/ephemeral/home/data/{lang}_receipt/ufo/train.json"
    img_path = f"/data/ephemeral/home/data/{lang}_receipt/img/"
    output_path = f"/data/ephemeral/home/data/{lang}_receipt/coco/train.json"
    
    # output 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"{lang} 데이터 변환 중...")
    convert_ufo_to_coco(ufo_path, img_path, output_path)
    print(f"{lang} 데이터 변환 완료")
        
