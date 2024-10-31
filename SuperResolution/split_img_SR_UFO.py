import json
import os
from PIL import Image
import numpy as np

def process_ufo_for_split_images(ufo_path, input_image_dir, output_image_dir, output_json_path):
    
    # 출력 디렉토리 생성
    os.makedirs(output_image_dir, exist_ok=True)

    # UFO 파일 로드
    with open(ufo_path, 'r', encoding='utf-8') as f:
        ufo_data = json.load(f)
    
    # 새로운 UFO 데이터 구조 생성
    new_ufo_data = {"images": {}}

    # 각 이미지 처리
    for img_name, img_data in ufo_data["images"].items():
        # 원본 이미지 크기 (4배 스케일 적용된 크기)
        width = img_data["img_w"]
        height = img_data["img_h"]
        
        # 크기 체크 (4000 이상 제외)
        if height <= 2000:
            scale = 4
        elif height <= 5000:
            scale = 2
        else:
            print(f"Skipping large image: {img_name} ({width}x{height})")
            continue

        # 분할 영역 정의
        if scale == 4:
            width = width * scale
            height = height * scale
            w_half = width // 2
            h_half = height // 2
            regions = [
                ((0, 0, w_half, h_half), "4_1"),
                ((w_half, 0, width, h_half), "4_2"),
                ((0, h_half, w_half, height), "4_3"),
                ((w_half, h_half, width, height), "4_4")
            ]

        elif scale == 2:
            width = width * scale
            height = height * scale
            w_half = width // 2
            h_half = height // 4
            regions = [
                ((0, 0, w_half, h_half), "8_1"),
                ((w_half, 0, width, h_half), "8_2"),
                ((0, h_half, w_half, h_half*2), "8_3"),
                ((w_half, h_half, width, h_half*2), "8_4"),
                ((0, h_half*2, w_half, h_half*3), "8_5"),
                ((w_half, h_half*2, width, h_half*3), "8_6"),
                ((0, h_half*3, w_half, height), "8_7"),
                ((w_half, h_half*3, width, height), "8_8")
            ]

        # 원본 이미지 로드
        img_path = os.path.join(input_image_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
            
        # 이미지 로드
        img = Image.open(img_path).convert('RGB')
        width, height = img.size

        # 분할 지점 계산
        w_half = width // 2
        h_half = height // 4
        
        
            
        # 각 분할 영역 처리
        for (x1, y1, x2, y2), position in regions:
            # 새 이미지 이름 생성
            new_img_name = f"{os.path.splitext(img_name)[0]}_{position}.jpg"

            # 이미지 분할
            cropped = img.crop((x1, y1, x2, y2))
            
            # 분할된 이미지 저장
            output_path = os.path.join(output_image_dir+'/x'+str(scale), new_img_name)
            cropped.save(output_path, 'JPEG', quality=95)
        
            # 새로운 words 데이터 생성
            new_words = {}
            
            # 각 word의 좌표 처리
            for word_id, word_data in img_data["words"].items():
                # 원본 좌표를 4배 스케일로 변환
                points = np.array(word_data["points"]) * 4
                
                # point가 현재 영역에 속하는지 확인
                in_region = True
                for point in points:
                    if not (x1 <= point[0] <= x2 and y1 <= point[1] <= y2):
                        in_region = False
                        break
                
                if in_region:
                    # 새로운 좌표 계산 (영역 내에서의 상대 좌표)
                    new_points = [
                        [p[0] - x1, p[1] - y1]
                        for p in points
                    ]
                    
                    # 새 word 데이터 추가
                    new_words[word_id] = {
                        "transcription": word_data["transcription"],
                        "points": new_points
                    }
            
            # words가 있는 경우에만 이미지 데이터 추가
            if new_words:
                new_ufo_data["images"][new_img_name] = {
                    "paragraphs": {},
                    "words": new_words,
                    "chars": {},
                    "img_w": x2 - x1,  # 분할된 영역의 너비
                    "img_h": y2 - y1,  # 분할된 영역의 높이
                    "num_patches": None,
                    "tags": img_data.get("tags", []),
                    "relations": {},
                    "annotation_log": img_data.get("annotation_log", {}),
                    "license_tag": img_data.get("license_tag", {})
                }
    
    # 새로운 UFO 파일 저장
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_ufo_data, f, ensure_ascii=False, indent=2)
    
    
    print(f"Original images: {len(ufo_data['images'])}")
    print(f"New images: {len(new_ufo_data['images'])}")

regions = ['chinese_receipt', 'japanese_receipt', 'thai_receipt', 'vietnamese_receipt']

# 사용 예시
for region in regions:
    ufo_path = "/data/ephemeral/home/data/"+region+"/ufo/train.json"
    input_image_dir = "/data/ephemeral/home/data/"+region+"/img/train"
    output_image_dir = "/data/ephemeral/home/JYP/level2-cv-datacentric-cv-14/SuperResolution/split_data/"+region+"/train"
    output_json_path = "/data/ephemeral/home/JYP/level2-cv-datacentric-cv-14/SuperResolution/split_data/"+region+"/train.json"



    process_ufo_for_split_images(ufo_path, input_image_dir, output_image_dir, output_json_path)
    print(f"Processed {region} UFO data saved to: {output_json_path}")