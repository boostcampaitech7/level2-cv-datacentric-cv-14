import os
import json
import argparse

def create_pseudo_labeled_json(args, countries):
    # CSV 파일 로드 및 모든 데이터를 메모리에 저장
    csv_path = args.csv
    with open(csv_path, 'r') as f:
        csv_data = json.load(f)["images"]
        
    # 각 나라별로 필터링하여 pseudo-labeled JSON 생성
    for country_code in countries:
        test_json_path = os.path.join(args.data_dir, country_code, "ufo", "test.json")
        output_json_path = os.path.join(args.data_dir, country_code, "ufo", "pseudo_train.json")
            
        # test.json 파일 로드
        with open(test_json_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            
        # 해당 국가에 맞는 이미지 ID만 필터링하여 points와 transcription 추가
        for image_id in test_data["images"].keys():
            if image_id in csv_data:
                for word_id, word_info in csv_data[image_id]["words"].items():
                    # 각 단어에 대해 points와 transcription 추가
                    test_data["images"][image_id]["words"][word_id] = {
                        "points": word_info["points"],
                        "transcription": "aa"  # transcription을 "aa"로 설정
                    }
            
        # pseudo-labeled JSON 파일로 저장
        with open(output_json_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(test_data, jsonfile, ensure_ascii=False, indent=4)
            
        print(f"Pseudo-labeled JSON saved to {output_json_path} for {country_code}")

countries = ["chinese_receipt", "thai_receipt", "japanese_receipt", "vietnamese_receipt"]  # 국가별 폴더명 리스트

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="/data/ephemeral/home/data", help="데이터가 존재하는 root 폴더")
parser.add_argument("--csv", default="./output.csv", help="pseudo labeling에 사용될 csv파일")
args = parser.parse_args()

create_pseudo_labeled_json(args, countries)