import json
import os

def merge_train_and_pseudo(data_dir, countries):
    for country_code in countries:
        train_json_path = os.path.join(data_dir, country_code, "ufo", "train_fold_1.json")
        pseudo_json_path = os.path.join(data_dir, country_code, "ufo", "pseudo_train.json")
        merged_json_path = os.path.join(data_dir, country_code, "ufo", "merged_train.json")
        
        # train.json 파일 로드
        with open(train_json_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        # pseudo_train.json 파일 로드
        with open(pseudo_json_path, 'r', encoding='utf-8') as f:
            pseudo_data = json.load(f)
        
        # train_data에 pseudo_data를 병합
        for image_id, image_info in pseudo_data["images"].items():
            # 같은 image_id가 train_data에 없을 경우 추가
            if image_id not in train_data["images"]:
                train_data["images"][image_id] = image_info
            else:
                # image_id가 train_data에 이미 있을 경우 words를 병합
                for word_id, word_info in image_info["words"].items():
                    if word_id not in train_data["images"][image_id]["words"]:
                        train_data["images"][image_id]["words"][word_id] = word_info
        
        # 병합된 데이터를 새로운 JSON 파일로 저장
        with open(merged_json_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=4)
        
        print(f"Merged train and pseudo-labeled data saved to {merged_json_path} for {country_code}")

# 사용 예시
data_dir = "/data/ephemeral/home/data"
countries = ["chinese_receipt", "thai_receipt", "japanese_receipt", "vietnamese_receipt"]  # 국가별 폴더명 리스트
merge_train_and_pseudo(data_dir, countries)
