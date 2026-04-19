"""
python main.py --mode train --train_tfrecord data/train_1014.tfrecord --val_tfrecord data/val_1014.tfrecord --epochs 30


python main.py --mode inference --val_tfrecord data/val_1014.tfrecord --checkpoint outputs/best_model.pt --output outputs/predictions.json


Verification indicators:
    - val_loss   total verification loss
    - avg_protagonist_ratio     On average, what proportion of segments per video are judged as protagonists
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_utils.loader import YT8MLoader
from src.models.multimodal_fusion import ProtagonistDetector
from src.models.loss_functions import ProtagonistLoss
from src.inference import run_inference



# TF Batch → PyTorch
def tf_batch_to_torch(batch: dict, device: torch.device):
    rgb    = torch.from_numpy(batch['rgb'].numpy()).float().to(device)
    audio  = torch.from_numpy(batch['audio'].numpy()).float().to(device)
    mask   = torch.from_numpy(batch['mask'].numpy()).bool().to(device)
    length = torch.from_numpy(batch['length'].numpy()).int().to(device)
    return rgb, audio, mask, length




def validate(model, criterion, val_ds, device):
    """
    Calculate loss and proxy metrics on validation set
    """
    model.eval()
    val_losses = []
    ratios     = []

    with torch.no_grad():
        for batch in val_ds:
            rgb, audio, mask, length = tf_batch_to_torch(batch, device)

            out = model(rgb, audio, mask, length)

            loss_dict = criterion(
                out, audio, mask, length,
                model.identity.protagonist_proto,
            )
            val_losses.append(loss_dict['loss'].item())

            # Proxy indicator: Proportion of relative protagonists in the video
            final_score = out['final_score'].cpu().numpy() 
            seg_mask    = out['seg_mask'].cpu().numpy()   
            B = final_score.shape[0]
            for b in range(B):
                valid  = seg_mask[b]
                n_valid = valid.sum()
                if n_valid < 2:
                    continue
                scores = final_score[b][valid]
                # Relative threshold: median within video
                median_thr = np.median(scores)
                n_prot = (scores > median_thr).sum()
                ratios.append(n_prot / n_valid)

    avg_loss  = float(np.mean(val_losses))  if val_losses else 0.0
    avg_ratio = float(np.mean(ratios))      if ratios     else 0.0
    return avg_loss, avg_ratio




def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[train] 设备: {device}')
    os.makedirs('outputs', exist_ok=True)

    # data
    print('load data')
    train_loader = YT8MLoader(
        args.train_tfrecord,
        batch_size=args.batch_size,
        shuffle=True,
        shuffle_buffer=500,
    )
    val_loader = YT8MLoader(
        args.val_tfrecord,
        batch_size=args.batch_size,
        shuffle=False,
    )
    train_ds = train_loader.get_dataset()
    val_ds   = val_loader.get_dataset()

    # model
    model = ProtagonistDetector(
        rgb_dim=1024, audio_dim=128,
        d_model=args.d_model,
        embed_dim=128,
        n_heads=8,
        n_layers=args.n_layers,
        num_segments=10,
        dropout=args.dropout,
        sync_weight=args.sync_weight,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'[train] 模型参数量: {total_params:,}')

    # loss function
    criterion = ProtagonistLoss(
        audio_dim=128,
        embed_dim=128,
        lambda_av=args.lambda_av,
        lambda_tc=args.lambda_tc,
        lambda_nce=args.lambda_nce,
        temperature=args.temperature,
    ).to(device)

    # optimizer
    # Prototype sets a larger learning rate separately to speed up convergence
    proto_params  = [model.identity.protagonist_proto]
    other_params  = [p for n, p in model.named_parameters()
                     if 'protagonist_proto' not in n]

    optimizer = optim.AdamW([
        {'params': other_params,  'lr': args.lr},
        {'params': proto_params,  'lr': args.lr * 5},
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # recover checkpoint
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch   = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f'[train] 从 epoch {start_epoch} 恢复，最佳 val_loss: {best_val_loss:.4f}')

    # train history
    history = {
        'train_loss': [], 'train_loss_av': [], 'train_loss_tc': [], 'train_loss_nce': [],
        'val_loss': [], 'val_protagonist_ratio': [],
    }

    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        ep_losses = {'loss': [], 'loss_av': [], 'loss_tc': [], 'loss_nce': []}
        t0 = time.time()
        step = 0

        for batch in train_ds:
            rgb, audio, mask, length = tf_batch_to_torch(batch, device)

            optimizer.zero_grad()

            out = model(rgb, audio, mask, length)
            loss_dict = criterion(
                out, audio, mask, length,
                model.identity.protagonist_proto,
            )

            loss = loss_dict['loss']
            loss.backward()

            # gradient crop
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        for k in ep_losses:
            ep_losses[k].append(
                loss_dict[k].item() if hasattr(loss_dict[k], 'item')
                else loss_dict[k]
            )

            # print
            if step % args.log_every == 0:
                print(f'  Epoch [{epoch+1:2d}/{args.epochs}] '
                      f'Step {step:4d} | '
                      f'loss={loss_dict["loss"].item():.4f}  '
                      f'av={loss_dict["loss_av"]:.4f}  '
                      f'tc={loss_dict["loss_tc"]:.4f}  '
                      f'nce={loss_dict["loss_nce"]:.4f}')
            step += 1

        scheduler.step()

        # calculate mean of epoch
        avg = {k: float(np.mean(v)) for k, v in ep_losses.items()}

        # validate
        val_loss, val_ratio = validate(
            model, criterion, val_ds, device,
        )

        elapsed = time.time() - t0
        print(f'\n[Epoch {epoch+1}/{args.epochs}]  '
              f'train_loss={avg["loss"]:.4f}  '
              f'val_loss={val_loss:.4f}  '
              f'protagonist_ratio={val_ratio:.3f}  '
              f'lr={scheduler.get_last_lr()[0]:.2e}  '
              f'({elapsed:.0f}s)\n')

        # record history
        history['train_loss'].append(avg['loss'])
        history['train_loss_av'].append(avg['loss_av'])
        history['train_loss_tc'].append(avg['loss_tc'])
        history['train_loss_nce'].append(avg['loss_nce'])
        history['val_loss'].append(val_loss)
        history['val_protagonist_ratio'].append(val_ratio)

        # save checkpoint
        ckpt = {
            'epoch'              : epoch,
            'model_state_dict'   : model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss'      : best_val_loss,
            'val_loss'           : val_loss,
            'val_ratio'          : val_ratio,
            'args'               : vars(args),
        }
        torch.save(ckpt, 'outputs/last_model.pt')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, 'outputs/best_model.pt')
            print(f'new best model (val_loss={best_val_loss:.4f})')

    # save history
    with open('outputs/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f'\ntraining complete. best val_loss: {best_val_loss:.4f}')
    print(f'checkpoint: outputs/best_model.pt')
    print(f'history:   outputs/training_history.json')




def inference(args):
    class InferArgs:
        tfrecord        = args.val_tfrecord
        checkpoint      = args.checkpoint
        output          = args.output
        batch_size      = args.batch_size
        d_model         = args.d_model
        n_layers        = args.n_layers
        n_clusters      = args.n_clusters
        top_ratio       = args.top_ratio

    run_inference(InferArgs())




def parse_args():
    p = argparse.ArgumentParser(description='protagonist detect')
    p.add_argument('--mode',             choices=['train', 'inference'], default='train')

    p.add_argument('--train_tfrecord',   type=str, default='data/train_1014.tfrecord')
    p.add_argument('--val_tfrecord',     type=str, default='data/val_1014.tfrecord')

    p.add_argument('--d_model',          type=int,   default=512)
    p.add_argument('--n_layers',         type=int,   default=2)
    p.add_argument('--dropout',          type=float, default=0.1)
    p.add_argument('--sync_weight',      type=float, default=0.4)

    p.add_argument('--epochs',           type=int,   default=30)
    p.add_argument('--batch_size',       type=int,   default=16)
    p.add_argument('--lr',               type=float, default=3e-4)
    p.add_argument('--lambda_av',        type=float, default=1.0)
    p.add_argument('--lambda_tc',        type=float, default=0.3)
    p.add_argument('--lambda_nce',       type=float, default=1.0)
    p.add_argument('--temperature',      type=float, default=0.07)
    p.add_argument('--log_every',        type=int,   default=20)

    
    p.add_argument('--resume',           type=str,   default=None)
    p.add_argument('--checkpoint',       type=str,   default='outputs/best_model.pt')

    
    p.add_argument('--output',           type=str,   default='outputs/predictions.json')
    p.add_argument('--n_clusters',       type=int,   default=5)
    p.add_argument('--top_ratio',        type=float, default=0.5)

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'train':
        train(args)
    else:
        inference(args)
