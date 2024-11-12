import os
import shutil
import argparse

def merge_images(data_dir, countries):
    # train과 test 이미지를 복사
    for country_code in countries:
        # 각 나라의 이미지 폴더 경로 설정
        train_dir = os.path.join(data_dir, country_code, "img", "train")
        test_dir = os.path.join(data_dir, country_code, "img", "test")
        merge_dir = os.path.join(data_dir, country_code, "img", "merge_image")

        # merge_image 폴더가 없으면 생성
        os.makedirs(merge_dir, exist_ok=True)

        # train 폴더에서 이미지 복사
        if os.path.exists(train_dir):
            for filename in os.listdir(train_dir):
                src_path = os.path.join(train_dir, filename)
                dst_path = os.path.join(merge_dir, filename)
                if os.path.isfile(src_path):  # 파일만 복사
                    shutil.copy(src_path, dst_path)

        # test 폴더에서 이미지 복사
        if os.path.exists(test_dir):
            for filename in os.listdir(test_dir):
                src_path = os.path.join(test_dir, filename)
                dst_path = os.path.join(merge_dir, filename)
                if os.path.isfile(src_path):  # 파일만 복사
                    shutil.copy(src_path, dst_path)

        print(f"Images from train and test copied to {merge_dir} for {country_code}")

# 사용 예시
countries = ["chinese_receipt", "thai_receipt", "japanese_receipt", "vietnamese_receipt"]  # 국가별 폴더명 리스트

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="/data/ephemeral/home/data", help="데이터가 존재하는 root 폴더")
args = parser.parse_args()

merge_images(args.data_dir, countries)