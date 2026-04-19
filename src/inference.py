"""
python src/inference.py --tfrecord data/val_1014.tfrecord --checkpoint outputs/best_model.pt --output outputs/predictions.json --n_clusters 5 --score_threshold 0.4

loader.py reads data
Collect embedding and final_score of all segments of each video
K-Means clustering (sklearn): find K identities in embedding space
The cluster with the most occurrences → the protagonist cluster
The segment corresponding to the protagonist cluster → converted to time interval (seconds)
Output JSON
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils.loader import YT8MLoader
from src.models.multimodal_fusion import ProtagonistDetector


def find_protagonist_cluster(
    embeds         : np.ndarray,   # L2 normalization embedding
    final_score    : np.ndarray,
    seg_mask       : np.ndarray,   # bool
    n_clusters     : int   = 5,
    top_ratio      : float = 0.5,  
    min_seg_ratio  : float = 0.1,  # at least 10%
    max_seg_ratio  : float = 0.7,  # at most 70%
) -> np.ndarray:
    """
    K-means
    Returns: bool
    """
    from sklearn.cluster import KMeans

    valid_idx = np.where(seg_mask)[0]
    n_valid   = len(valid_idx)
    is_protagonist = np.zeros(len(seg_mask), dtype=bool)

    if n_valid == 0:
        return is_protagonist

    scores_valid = final_score[valid_idx] 
    embeds_valid = embeds[valid_idx] 

    # Relative threshold within video
    n_top = int(np.clip(
        round(n_valid * top_ratio),
        max(1, round(n_valid * min_seg_ratio)),
        round(n_valid * max_seg_ratio),
    ))
    score_rank   = np.argsort(scores_valid)[::-1]
    # Local index of top n_top scores
    top_local    = set(score_rank[:n_top]) 

    # K-Means clusters
    k = min(n_clusters, n_valid)
    if k < 2:
        # only 1 valid fragment
        is_protagonist[valid_idx[0]] = True
        return is_protagonist

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeds_valid)

    
    cluster_count = np.bincount(labels, minlength=k)
    cluster_score = np.array([
        scores_valid[labels == c].mean() if (labels == c).sum() > 0 else 0.0
        for c in range(k)
    ])
    protagonist_cluster = np.lexsort((-cluster_score, -cluster_count))[0]

    # is protagonist clusters and has a relatively high score
    for local_i, (global_idx, lbl) in enumerate(zip(valid_idx, labels)):
        if lbl == protagonist_cluster and local_i in top_local:
            is_protagonist[global_idx] = True

    return is_protagonist



def segments_to_intervals(
    is_protagonist : np.ndarray, 
    final_score    : np.ndarray, 
    video_length   : int,       
    num_segments   : int = 10,
) -> list:
    """
    segment index → time interval
    Merge consecutive main segments into an interval
    """
    S = len(is_protagonist)
    
    boundaries = np.linspace(0, video_length, S + 1)

    intervals = []
    in_seg    = False
    seg_start = 0.0
    seg_confs = []

    for s in range(S):
        if is_protagonist[s]:
            if not in_seg:
                in_seg    = True
                seg_start = boundaries[s]
                seg_confs = []
            seg_confs.append(float(final_score[s]))
        else:
            if in_seg:
                in_seg = False
                intervals.append({
                    'start_sec' : round(float(seg_start), 1),
                    'end_sec'   : round(float(boundaries[s]), 1),
                    'confidence': round(float(np.mean(seg_confs)), 4),
                })

    if in_seg:
        intervals.append({
            'start_sec' : round(float(seg_start), 1),
            'end_sec'   : round(float(boundaries[S]), 1),
            'confidence': round(float(np.mean(seg_confs)), 4),
        })

    return intervals



def tf_batch_to_torch(batch: dict, device: torch.device):
    """
    TF Batch → PyTorch Tensor
    """
    rgb    = torch.from_numpy(batch['rgb'].numpy()).float().to(device)
    audio  = torch.from_numpy(batch['audio'].numpy()).float().to(device)
    mask   = torch.from_numpy(batch['mask'].numpy()).bool().to(device)
    length = torch.from_numpy(batch['length'].numpy()).int().to(device)
    video_ids = [v.decode('utf-8') if isinstance(v, bytes) else str(v)
                 for v in batch['video_id'].numpy()]
    return rgb, audio, mask, length, video_ids



def run_inference(args) -> dict:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    model = ProtagonistDetector(
        rgb_dim=1024, audio_dim=128,
        d_model=args.d_model, embed_dim=128,
        n_heads=8, n_layers=args.n_layers,
        num_segments=10,
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        epoch = ckpt.get('epoch', '?')
        print(f'load checkpoint (epoch {epoch}): {args.checkpoint}')
    else:
        print('lost checkpoint, use random weight')

    model.to(device)
    model.eval()

    loader = YT8MLoader(
        args.tfrecord,
        batch_size=args.batch_size,
        shuffle=False,
    )
    dataset = loader.get_dataset()

    predictions = {}
    total = 0
    t0    = time.time()

    with torch.no_grad():
        for batch in dataset:
            rgb, audio, mask, length, video_ids = tf_batch_to_torch(batch, device)

            out = model(rgb, audio, mask, length)

            embeds      = out['embeds'].cpu().numpy()
            final_score = out['final_score'].cpu().numpy()
            seg_mask    = out['seg_mask'].cpu().numpy()
            sync_score  = out['sync_score'].cpu().numpy()
            proto_score = out['proto_score'].cpu().numpy()
            lengths_np  = length.cpu().numpy()

            B = len(video_ids)
            for i in range(B):
                vid   = video_ids[i]
                L     = int(lengths_np[i])

                # K-Means + Relative threshold within video
                is_protagonist = find_protagonist_cluster(
                    embeds[i], final_score[i], seg_mask[i],
                    n_clusters=args.n_clusters,
                    top_ratio=args.top_ratio,
                )

                # segment → time interval
                intervals = segments_to_intervals(
                    is_protagonist, final_score[i], L, num_segments=10
                )

                # segment score
                seg_details = []
                boundaries = np.linspace(0, L, 11)
                for s in range(10):
                    if not seg_mask[i][s]:
                        continue
                    seg_details.append({
                        'segment'     : s,
                        'start_sec'   : round(float(boundaries[s]),   1),
                        'end_sec'     : round(float(boundaries[s+1]), 1),
                        'final_score' : round(float(final_score[i][s]),  4),
                        'proto_score' : round(float(proto_score[i][s]),  4),
                        'sync_score'  : round(float(sync_score[i][s]),   4),
                        'is_protagonist': bool(is_protagonist[s]),
                    })

                predictions[vid] = {
                    'video_length_sec'    : L,
                    'protagonist_segments': intervals,
                    'has_protagonist'     : len(intervals) > 0,
                    'segment_details'     : seg_details,
                }
                total += 1

            if total % 50 == 0 and total > 0:
                print(f'{total} done')

    elapsed = time.time() - t0
    n_with  = sum(1 for v in predictions.values() if v['has_protagonist'])
    print(f'\ncomplete, {total} videos, use {elapsed:.1f}s')
    print(f'detect protagonist: {n_with}/{total} ({100*n_with/max(total,1):.1f}%)')

    # save JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f'save: {args.output}')

    return predictions



def parse_args():
    p = argparse.ArgumentParser(description='Protagonist detection inference')
    p.add_argument('--tfrecord',         type=str,   required=True)
    p.add_argument('--checkpoint',       type=str,   default=None)
    p.add_argument('--output',           type=str,   default='outputs/predictions.json')
    p.add_argument('--batch_size',       type=int,   default=16)
    p.add_argument('--d_model',          type=int,   default=512)
    p.add_argument('--n_layers',         type=int,   default=2)
    p.add_argument('--n_clusters',       type=int,   default=5)
    p.add_argument('--top_ratio',        type=float, default=0.5)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_inference(args)