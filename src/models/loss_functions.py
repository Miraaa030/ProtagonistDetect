"""

L1  AudioVisualAlignmentLoss -- Papalampidi and Lapata (2023)
    For segments with vocals, the cosine similarity between the visual embedding and the audio embedding 
    is maximized. Make audio and video features highly consistent


L2  TemporalConsistencyLoss -- Zheng et al. (2015) Re-ID robustness
    final_score changes of the protagonists in adjacent segments should be as smooth as possible 
    and will not disappear/appear suddenly between adjacent shots.


L3  InfoNCEPrototypeLoss
    Within each video, the similarity between the top-k segments (positive samples) and 
    the prototype is forced to be significantly higher than that of the bot-k segments (negative samples), 
    preventing the prototype from drifting to the global mean.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class AudioVisualAlignmentLoss(nn.Module):
    """
    L1
    Project the audio segment features into the same embed_dim space as the visual embedding;
    Calculate the cosine similarity between audio_embed and visual_embed for each segment;
    Use the syncscore of the segment as the weight, Loss = -weighted cosine similarity mean
    """

    def __init__(self, audio_dim: int = 128, embed_dim: int = 128):
        super().__init__()
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        audio        : torch.Tensor,
        visual_embed : torch.Tensor, 
        sync_score   : torch.Tensor, 
        mask         : torch.Tensor, 
        length       : torch.Tensor, 
        seg_mask     : torch.Tensor, 
    ) -> torch.Tensor:

        B, T, _ = audio.shape
        S = visual_embed.size(1)

        # Audio segment level pooling
        audio_seg = torch.zeros(B, S, audio.size(-1), device=audio.device)
        for b in range(B):
            L = length[b].item()
            if L == 0:
                continue
            boundaries = torch.linspace(0, L, S + 1).long()
            for s in range(S):
                start = boundaries[s].item()
                end   = boundaries[s + 1].item()
                if start >= end:
                    continue
                seg_m = mask[b, start:end]
                if seg_m.sum() == 0:
                    continue
                vals = audio[b, start:end]
                audio_seg[b, s] = (vals * seg_m.unsqueeze(-1).float()).sum(0) / seg_m.float().sum()

        # Project and normalize
        B_S = B * S
        audio_proj = self.audio_proj(audio_seg.view(B_S, -1))   # [B*S, embed_dim]
        audio_proj = F.normalize(audio_proj, p=2, dim=-1).view(B, S, -1)

        cos_sim = (visual_embed * audio_proj).sum(dim=-1)

        # WEIGHTED
        weight = sync_score * seg_mask.float()
        weight_sum = weight.sum().clamp(min=1e-6)

        loss = -(cos_sim * weight).sum() / weight_sum
        return loss



class TemporalConsistencyLoss(nn.Module):
    """
    L2
    Average the absolute values of final_score differences between adjacent valid segments
    """

    def forward(
        self,
        final_score: torch.Tensor,
        seg_mask   : torch.Tensor,
    ) -> torch.Tensor:

        if final_score.size(1) < 2:
            return torch.tensor(0.0, device=final_score.device)

        # 相邻段均有效才参与计算
        both_valid = seg_mask[:, 1:] & seg_mask[:, :-1] 
        diff = (final_score[:, 1:] - final_score[:, :-1]).abs() 

        n = both_valid.float().sum().clamp(min=1e-6)
        return (diff * both_valid.float()).sum() / n



class InfoNCEPrototypeLoss(nn.Module):
    """
    L3

    Relative sorting within the video determines positive and negative samples, 
    and each positive sample is compared with all negative samples.
    The temperature parameter T controls the discrimination
    """

    def __init__(self, top_k: int = 3, bot_k: int = 3, temperature: float = 0.07):
        super().__init__()
        self.top_k       = top_k
        self.bot_k       = bot_k
        self.temperature = temperature

    def forward(
        self,
        embeds      : torch.Tensor,
        final_score : torch.Tensor, 
        seg_mask    : torch.Tensor, 
        prototype   : torch.Tensor,
    ) -> torch.Tensor:

        B, S, D = embeds.shape
        proto = F.normalize(prototype, p=2, dim=0)

        video_losses = []

        for b in range(B):
            valid_idx = seg_mask[b].nonzero(as_tuple=True)[0]
            n_valid   = valid_idx.numel()

            # at least 2 valid fragment
            if n_valid < 2:
                continue

            scores_valid = final_score[b][valid_idx] 
            embeds_valid = embeds[b][valid_idx]

            # sort
            sorted_idx = scores_valid.argsort(descending=True)

            k_pos = min(self.top_k, n_valid // 2)
            k_neg = min(self.bot_k, n_valid - k_pos)

            if k_pos == 0 or k_neg == 0:
                continue

            pos_emb = embeds_valid[sorted_idx[:k_pos]]
            neg_emb = embeds_valid[sorted_idx[n_valid-k_neg:]]

            # calculate InfoNCE loss for each pos
            # anchor = prototype
            # (prototype, pos_emb), (prototype, neg_emb)
            #logits = [sim(proto,pos_0), sim(proto,pos_1), ..., sim(proto,neg_0), sim(proto,neg_1), ...]  / T
            # label  = The position of the positive sample in each row（0..k_pos-1）

            pos_sim = (pos_emb * proto.unsqueeze(0)).sum(-1) / self.temperature
            neg_sim = (neg_emb * proto.unsqueeze(0)).sum(-1) / self.temperature

            # 
            neg_sim_expand = neg_sim.unsqueeze(0).expand(k_pos, -1)
            logits = torch.cat([
                pos_sim.unsqueeze(1),
                neg_sim_expand, 
            ], dim=1) 

            # Positive samples in each row are in column 0
            labels = torch.zeros(k_pos, dtype=torch.long, device=embeds.device)
            loss_b = F.cross_entropy(logits, labels)
            video_losses.append(loss_b)

        if not video_losses:
            return torch.tensor(0.0, requires_grad=True, device=embeds.device)

        return torch.stack(video_losses).mean()



class ProtagonistLoss(nn.Module):
    """
    L_total = λ1·L_av + λ2·L_tc + λ3·L_nce
    """

    def __init__(
        self,
        audio_dim   : int   = 128,
        embed_dim   : int   = 128,
        lambda_av   : float = 1.0,
        lambda_tc   : float = 0.3,
        lambda_nce  : float = 1.0,
        top_k       : int   = 3,
        bot_k       : int   = 3,
        temperature : float = 0.07,
    ):
        super().__init__()
        self.lambda_av  = lambda_av
        self.lambda_tc  = lambda_tc
        self.lambda_nce = lambda_nce

        self.av_loss  = AudioVisualAlignmentLoss(audio_dim=audio_dim, embed_dim=embed_dim)
        self.tc_loss  = TemporalConsistencyLoss()
        self.nce_loss = InfoNCEPrototypeLoss(
            top_k=top_k, bot_k=bot_k, temperature=temperature
        )

    def forward(
        self,
        model_output : Dict[str, torch.Tensor],
        audio        : torch.Tensor, 
        mask         : torch.Tensor, 
        length       : torch.Tensor, 
        prototype    : torch.Tensor, 
    ) -> Dict[str, torch.Tensor]:

        embeds      = model_output['embeds'] 
        final_score = model_output['final_score'] 
        sync_score  = model_output['sync_score'] 
        seg_mask    = model_output['seg_mask'] 

        l_av = self.av_loss(
            audio=audio,
            visual_embed=embeds,
            sync_score=sync_score,
            mask=mask,
            length=length,
            seg_mask=seg_mask,
        )

        l_tc = self.tc_loss(final_score, seg_mask)

        l_nce = self.nce_loss(embeds, final_score, seg_mask, prototype)

        total = (
            self.lambda_av * l_av + self.lambda_tc * l_tc + self.lambda_nce * l_nce
        )

        return {
            'loss'     : total,
            'loss_av'  : l_av.item(),
            'loss_tc'  : l_tc.item(),
            'loss_nce' : l_nce.item(),
        }



if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    import torch

    torch.manual_seed(42)

    B, T, S = 4, 284, 10
    lengths  = torch.tensor([245, 132, 284, 133])

    rgb   = torch.rand(B, T, 1024)
    audio = torch.rand(B, T, 128)
    mask  = torch.zeros(B, T, dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = True

    from src.models.multimodal_fusion import ProtagonistDetector
    model = ProtagonistDetector()
    model.train()

    out = model(rgb, audio, mask, lengths)

    criterion = ProtagonistLoss(temperature=0.07)
    prototype = model.identity.protagonist_proto

    losses = criterion(out, audio, mask, lengths, prototype)

    print('loss:')
    for k, v in losses.items():
        val = v.item() if hasattr(v, 'item') else v
        print(f'  {k:10s}: {val:.6f}')

    # valid gradient
    losses['loss'].backward()
    proto_grad = model.identity.protagonist_proto.grad
    print(f'\n  prototype.grad norm: {proto_grad.norm():.6f}  (> 0)')
    print('\n  loss_nce >=0 (InfoNCE initial ≈ 1.4)')