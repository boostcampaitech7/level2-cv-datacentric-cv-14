import os
import json
import argparse

def merge_train_and_pseudo(args, countries):
    # train과 pseudo json파일을 합치는 함수
    for country_code in countries:
        train_json_path = os.path.join(args.data_dir, country_code, "ufo", args.train)
        pseudo_json_path = os.path.join(args.data_dir, country_code, "ufo", args.pseudo)
        merged_json_path = os.path.join(args.data_dir, country_code, "ufo", "merged_train.json")
        
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
countries = ["chinese_receipt", "thai_receipt", "japanese_receipt", "vietnamese_receipt"]  # 국가별 폴더명 리스트

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="/data/ephemeral/home/data", help="데이터가 존재하는 root 폴더")
parser.add_argument("--train", default="train.json", help="train에 사용되는 json 파일명")
parser.add_argument("--pseudo", default="pseudo_train.json", help="pseudo labeling을 통해서 나온 json 파일명")
args = parser.parse_args()

merge_train_and_pseudo(args, countries)
