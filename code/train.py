import os
import os.path as osp
import time
import math
import wandb
import numpy as np
from datetime import timedelta
from argparse import ArgumentParser, Namespace
from omegaconf import OmegaConf

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset, CustomDataset
from model import EAST
from utils.Gsheet import Gsheet_param
from utils.wandb import set_wandb
import albumentations as A



def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, validation, train_ann, val_ann, custom_transform):
    # 1. 훈련 데이터셋 로드 및 전처리 
    # train_dataset = SceneTextDataset(
    #     data_dir,
    #     split=train_ann,
    #     image_size=image_size,
    #     crop_size=input_size,
    #     validation=False
    # )

    train_dataset = CustomDataset(
        data_dir,
        split=train_ann,
        transform=custom_transform
    )
    
    train_dataset = EASTDataset(train_dataset)
    train_num_batches = math.ceil(len(train_dataset) / batch_size)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    #2. 검증 데이터셋 로드 및 전처리 ( Validation = True일 때 사용함)
    if validation:
        valid_dataset = SceneTextDataset(
            data_dir,
            split=val_ann,
            image_size=image_size,
            crop_size=input_size,
            validation=True
        )
        valid_dataset = EASTDataset(valid_dataset)
        valid_num_batches = math.ceil(len(valid_dataset) / batch_size)
        
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    # 3. 모델 초기화 및 학습 설정 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)
     
    # Early stopping 설정 변수 
    counter = 0
    best_val_loss = np.inf
    
    # 4. 훈련 단계 
    model.train()
    for epoch in range(max_epoch):
        train_loss, valid_loss = 0, 0
        train_start = time.time()
        cls_loss_total, angle_loss_total, iou_loss_total = 0, 0, 0  # 각 손실 항목 누적 변수
        
        # 훈련 진행률 표시 
        with tqdm(total=train_num_batches, desc=f'[Training Epoch {epoch + 1}]', disable=False) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 손실 값 계산 및 누적
                loss_val = loss.item()
                train_loss += loss_val
                cls_loss_total += extra_info['cls_loss']
                angle_loss_total += extra_info['angle_loss']
                iou_loss_total += extra_info['iou_loss']
                
                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }

                pbar.set_postfix(val_dict)
        
        # 에폭이 종료 후 손실 평균값 'wandb'에 기록 
        wandb.log({
            "Epochs": epoch + 1,
            "Train Mean loss": train_loss / train_num_batches,
            "Cls loss": cls_loss_total / train_num_batches,
            "Angle loss": angle_loss_total / train_num_batches,
            "IoU loss": iou_loss_total / train_num_batches,
            "Learning Rate": scheduler.get_last_lr()[0],
        })
        train_end = time.time() - train_start
        print("Train Mean loss: {:.4f} || Elapsed time: {} || ETA: {}".format(
            train_loss / train_num_batches,
            timedelta(seconds=train_end),
            timedelta(seconds=train_end*(max_epoch-epoch+1))))
            
        #---------------------validation---------------------#    
        if validation :
            model.eval()
            with torch.no_grad():
                valid_start = time.time()
                print("Evaluating validation results...")
                valid_cls_loss_total, valid_angle_loss_total, valid_iou_loss_total = 0, 0, 0  # 검증 손실 누적 변수
                
                #검증 진행률 표시 
                with tqdm(total=valid_num_batches, desc=f'[Validation Epoch {epoch + 1}]', disable=False) as pbar:
                    for idx, (img, gt_score_map, gt_geo_map, roi_mask) in enumerate(valid_loader):
                        loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                        
                        #손실 값 누적 
                        loss_val = loss.item()
                        valid_loss += loss_val
                        valid_cls_loss_total += extra_info['cls_loss']
                        valid_angle_loss_total += extra_info['angle_loss']
                        valid_iou_loss_total += extra_info['iou_loss']
                        
                        pbar.update(1)
                        val_dict = {
                            'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                            'IoU loss': extra_info['iou_loss']
                        }
                        pbar.set_postfix(val_dict)
                    
                # Validation 평균 손실 계산 및 로깅
                wandb.log({
                    "Valid Mean loss": valid_loss / valid_num_batches,
                    "Valid Cls loss": valid_cls_loss_total / valid_num_batches,
                    "Valid Angle loss": valid_angle_loss_total / valid_num_batches,
                    "Valid IoU loss": valid_iou_loss_total / valid_num_batches,
                })
                    
                if not osp.exists(model_dir):
                    os.makedirs(model_dir)
                
                # Best Model 저장 로직( 손실 값이 개선된 경우에만 저장함)
                mean_val_loss = valid_loss / valid_num_batches
                if best_val_loss > mean_val_loss:
                    best_val_loss = mean_val_loss
                    best_val_loss_epoch = epoch+1
                    ckpt_fpath = osp.join(model_dir, f"best_epoch_{best_val_loss_epoch}.pth")
                    torch.save(model.state_dict(), ckpt_fpath)
                    counter = 0
                else:
                    counter += 1
                    print(f"Not Val Update.. Counter : {counter}")
                    
            valid_end = time.time() - valid_start
            print("Valid Mean loss: {:.4f} || Elapsed time: {}".format(
                mean_val_loss,
                timedelta(seconds=valid_end)))
            
            print("Best Validation Loss: {:.4f} at Epoch {}".format(
                best_val_loss,
                best_val_loss_epoch))

        # Validation == False 
        else:
            if (epoch + 1) >= 100 and (epoch + 1) % save_interval == 0:
                if not osp.exists(model_dir):
                    os.makedirs(model_dir)
                ckpt_fpath = osp.join(model_dir, f"epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), ckpt_fpath)
        print("")
        # 학습률 스케줄러 업데이트 
        scheduler.step()    


def main(args):
    args_dict = OmegaConf.to_container(args, resolve=True)
    set_wandb(args.experiment_name, args.experiment_detail, args)
    training_args = {
        k: v for k, v in args_dict.items() 
        if k in do_training.__code__.co_varnames
    }

    training_args['custom_transform'] = [getattr(A, aug)(**params) 
                                         for aug, params in args_dict['transform'].items()]

    args = Namespace(**training_args)
    do_training(**args.__dict__)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/base_train.yaml"
    )
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)
    main(cfg)
    Gsheet_param(cfg)