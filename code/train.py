import os
import os.path as osp
import time
import math
import wandb
from datetime import timedelta
from argparse import ArgumentParser, Namespace
from omegaconf import OmegaConf

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST
from utils.Gsheet import Gsheet_param
from utils.wandb import set_wandb



def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval):
    dataset = SceneTextDataset(
        data_dir,
        split='train',
        image_size=image_size,
        crop_size=input_size,
    )
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    model.train()
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                wandb.log({
                    "train_loss": loss,
                    "train_cls_loss": val_dict['Cls loss'],
                    "train_angle_loss": val_dict['Angle loss'],
                    "train_IoU_loss": val_dict['Cls loss'],
                    'Epoch': epoch
                }, step=epoch+1)
                pbar.set_postfix(val_dict)

        scheduler.step()

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)


def main(args):
    args_dict = OmegaConf.to_container(args, resolve=True)
    set_wandb(args.experiment_name, args.experiment_detail, args)
    training_args = {
        k: v for k, v in args_dict.items() 
        if k in do_training.__code__.co_varnames
    }
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