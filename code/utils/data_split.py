import json
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
import os

def stratified_kfold_split(data, n_splits=5, seed=42, save_dir='/data/ephemeral/home/data/chinese_receipt/ufo'):
    # 이미지 ID와 바운딩 박스 개수 리스트 생성
    # 이미지 ID 목록 생성
    image_ids = list(data['images'].keys())
    
    # KFold 객체 초기화
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    
    # Fold별로 데이터 분할
    for fold, (train_idx, val_idx) in enumerate(kf.split(image_ids), start=1):
        train_data = {"images": {}}
        val_data = {"images": {}}
        
        # Train set에 해당하는 이미지 저장
        for idx in train_idx:
            img_id = image_ids[idx]
            train_data["images"][img_id] = data["images"][img_id]
        
        # Validation set에 해당하는 이미지 저장
        for idx in val_idx:
            img_id = image_ids[idx]
            val_data["images"][img_id] = data["images"][img_id]
        
        # JSON 파일로 저장
        train_path = os.path.join(save_dir, f"train_fold_{fold}.json")
        val_path = os.path.join(save_dir, f"val_fold_{fold}.json")
        
        with open(train_path, 'w') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=4)
        
        with open(val_path, 'w') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=4)
        
        print(f"Fold {fold} saved to {train_path} and {val_path}")


country = ['chinese_receipt', 'japanese_receipt', 'thai_receipt', 'vietnamese_receipt']
for i in range(4) :
    # 데이터 로드
    with open(f"/data/ephemeral/home/data/{country[i]}/ufo/train.json", "r") as file:
        data = json.load(file)

    # Stratified K-Fold로 데이터 분할 및 저장
    stratified_kfold_split(data, n_splits=5, seed=42, save_dir=f'/data/ephemeral/home/data/{country[i]}/ufo')
