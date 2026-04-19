"""
input(torch):
    rgb   : [B, T, 1024]
    audio : [B, T, 128]
    mask  : [B, T]
    length: [B] 

Pipeline:
    CrossModalAdapter:          Audio guides visual attention   [B, T, d_model]
    AdaptiveHierarchicalPool:   Adaptive 10-segment pooling     [B, 10, d_model]
    SegmentTransformer:         Cross-segment timing reasoning  [B, 10, d_model]
    ]IdentityHead:              Re-ID Embedding + Protagonist Score

Protagonist score = mean audio-visual synchronization weight
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class CrossModalAdapter(nn.Module):
    """
    input:
        rgb   : [B, T, 1024]
        audio : [B, T, 128]
        mask  : [B, T]  bool, False is padding
    output:
        fused : [B, T, d_model]
        attn_w: [B, T]
    """

    def __init__(
        self,
        rgb_dim  : int = 1024,
        audio_dim: int = 128,
        d_model  : int = 512,
        n_heads  : int = 8,
        dropout  : float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # project
        self.rgb_proj   = nn.Linear(rgb_dim,   d_model)
        self.audio_proj = nn.Linear(audio_dim, d_model)

        # bull cross attention: Q=RGB，K/V=Audio
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Bottleneck Adapter
        mid = d_model // 4
        self.adapter = nn.Sequential(
            nn.Linear(d_model, mid),
            nn.GELU(),
            nn.Linear(mid, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(
        self,
        rgb  : torch.Tensor,
        audio: torch.Tensor, 
        mask : torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # nn.MultiheadAttention key_padding_mask：True = ignore
        # negate
        key_pad_mask = ~mask  

        q = self.rgb_proj(rgb)   
        k = self.audio_proj(audio) 
        v = k

        attn_out, attn_w = self.cross_attn(
            q, k, v,
            key_padding_mask=key_pad_mask,
            need_weights=True,
            average_attn_weights=True,   
        )

        # residual + LayerNorm
        x = self.norm1(q + self.drop(attn_out))
        x = self.norm2(x + self.adapter(x)) 

        # Audio and video synchronization weight of each frame: summation of key dimensions
        # attn_w: [B, T, T]
        sync_score = attn_w.sum(dim=-1) 
        # put 0
        sync_score = sync_score * mask.float()

        return x, sync_score



class AdaptiveHierarchicalPool(nn.Module):
    """
    Adaptively divide the variable-length time series [B, T, D] into num_segments segments, 
    and dynamically calculate segment boundaries based on the real length of the video.
    """

    def __init__(self, num_segments: int = 10):
        super().__init__()
        self.num_segments = num_segments

    def forward(
        self,
        x     : torch.Tensor, 
        mask  : torch.Tensor,  
        length: torch.Tensor,  
    # Returns: seg_feat: [B, num_segments, D]  seg_mask: [B, num_segments] 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        

        B, T, D = x.shape
        S = self.num_segments
        device = x.device

        seg_feat = torch.zeros(B, S, D, device=device, dtype=x.dtype)
        seg_mask = torch.zeros(B, S,    device=device, dtype=torch.bool)

        for b in range(B):
            L = length[b].item()
            if L == 0:
                continue

            # split
            boundaries = torch.linspace(0, L, S + 1).long()

            for s in range(S):
                start = boundaries[s].item()
                end   = boundaries[s + 1].item()
                if start >= end:
                    continue

                seg_frames = x[b, start:end, :]
                seg_m      = mask[b, start:end]

                if seg_m.sum() == 0:
                    continue

                # Masked mean pooling
                seg_feat[b, s] = (seg_frames * seg_m.unsqueeze(-1).float()).sum(0) \
                                 / seg_m.float().sum()
                seg_mask[b, s] = True

        return seg_feat, seg_mask



class SegmentTransformer(nn.Module):
    """
    Perform global self-attention on 10 segments to learn the long-term appearance pattern 
    of the protagonist across shots.
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True, 
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pos_emb  = nn.Embedding(50, d_model) 

    def forward(
        self,
        x        : torch.Tensor,  
        seg_mask : torch.Tensor, 
    ) -> torch.Tensor:

        S = x.size(1)
        pos = torch.arange(S, device=x.device)
        x   = x + self.pos_emb(pos).unsqueeze(0) 

        # src_key_padding_mask：True=ignore
        pad_mask = ~seg_mask 
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return x 



class IdentityHead(nn.Module):
    """
    input:  [B, S, d_model]
    output:
        embed: [B, S, embed_dim]  L2 normalized identity vector (for K-Means clustering)
        proto_score: [B, S]
    """

    def __init__(self, d_model: int = 512, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # CEA-style Nonlinear dimensionality reduction
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, embed_dim),
        )

        # Learnable protagonist prototype vector
        self.protagonist_proto = nn.Parameter(
            torch.randn(embed_dim) * 0.01
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, _ = x.shape

        embed = self.proj(x.view(B * S, -1)) 
        embed = F.normalize(embed, p=2, dim=-1) 
        embed = embed.view(B, S, self.embed_dim) 

        # Cosine similarity to the protagonist prototype -> protagonist score (0~1)
        proto = F.normalize(self.protagonist_proto, p=2, dim=0)
        proto_score = (embed * proto.unsqueeze(0).unsqueeze(0)).sum(-1) 
        # Map from [-1,1] to [0,1]
        proto_score = (proto_score + 1) / 2 

        return embed, proto_score



class ProtagonistDetector(nn.Module):
    """
    input:
        rgb   : [B, T, 1024]
        audio : [B, T, 128]
        mask  : [B, T] 
        length: [B] 

    output dict:
        'embeds'     : [B, 10, 128]  Re-ID embedding for each segment
        'proto_score': [B, 10]       Protagonist prototype similarity in each paragraph
        'sync_score' : [B, 10]       The average audio and video synchronization weight of each segment
        'final_score': [B, 10]       Comprehensive protagonist score
        'seg_mask'   : [B, 10]       valid / invalid
    """

    def __init__(
        self,
        rgb_dim     : int   = 1024,
        audio_dim   : int   = 128,
        d_model     : int   = 512,
        embed_dim   : int   = 128,
        n_heads     : int   = 8,
        n_layers    : int   = 2,
        num_segments: int   = 10,
        dropout     : float = 0.1,
        # The proportion of audiovisual synchronization weight in the comprehensive score
        sync_weight : float = 0.4,
    ):
        super().__init__()
        self.num_segments = num_segments
        self.sync_weight  = sync_weight

        self.cross_modal = CrossModalAdapter(
            rgb_dim=rgb_dim, audio_dim=audio_dim,
            d_model=d_model, n_heads=n_heads, dropout=dropout,
        )
        self.hier_pool = AdaptiveHierarchicalPool(num_segments=num_segments)
        self.seg_transformer = SegmentTransformer(
            d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, dropout=dropout,
        )
        self.identity = IdentityHead(
            d_model=d_model, embed_dim=embed_dim, dropout=dropout,
        )

    def forward(
        self,
        rgb   : torch.Tensor,
        audio : torch.Tensor, 
        mask  : torch.Tensor, 
        length: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        
        fused, sync_frame = self.cross_modal(rgb, audio, mask)
        
        seg_feat, seg_mask = self.hier_pool(fused, mask, length)
        
        sync_seg = self._pool_sync(sync_frame, mask, length)
       
        seg_out = self.seg_transformer(seg_feat, seg_mask)
       
        embeds, proto_score = self.identity(seg_out)
       


        # Overall protagonist score
        # Normalize sync_seg to 0~1
        sync_norm = self._minmax_norm(sync_seg, seg_mask)

        final_score = (1 - self.sync_weight) * proto_score \
                    +      self.sync_weight  * sync_norm
        # Invalid segments are set to 0
        final_score = final_score * seg_mask.float()

        return {
            'embeds'     : embeds,      
            'proto_score': proto_score,  
            'sync_score' : sync_norm,  
            'final_score': final_score, 
            'seg_mask'   : seg_mask, 
        }

    

    def _pool_sync(
        self,
        sync_frame: torch.Tensor,   
        mask      : torch.Tensor,  
        length    : torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate frame-level sync weights to segment level
        """
        B, T = sync_frame.shape
        S    = self.num_segments
        sync_seg = torch.zeros(B, S, device=sync_frame.device)

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
                vals = sync_frame[b, start:end]
                sync_seg[b, s] = (vals * seg_m.float()).sum() / seg_m.float().sum()

        return sync_seg

    @staticmethod
    def _minmax_norm(
        x   : torch.Tensor,  
        mask: torch.Tensor, 
    ) -> torch.Tensor:
        """Perform min-max normalization on each video independently, 
        and the padding segment is not involved.
        """
        B, S = x.shape
        out  = torch.zeros_like(x)
        for b in range(B):
            valid = x[b][mask[b]]
            if valid.numel() == 0:
                continue
            mn, mx = valid.min(), valid.max()
            denom  = (mx - mn).clamp(min=1e-6)
            out[b] = ((x[b] - mn) / denom) * mask[b].float()
        return out



if __name__ == '__main__':
    import numpy as np

    torch.manual_seed(0)
    model = ProtagonistDetector()
    model.eval()

    B, T = 4, 284
    lengths = torch.tensor([245, 132, 284, 133])

    rgb   = torch.rand(B, T, 1024)
    audio = torch.rand(B, T, 128)


    mask = torch.zeros(B, T, dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = True

    print('input:')
    print(f'  rgb   : {rgb.shape}')
    print(f'  audio : {audio.shape}')
    print(f'  mask  : {mask.shape}')
    print(f'  length: {lengths.tolist()}')

    with torch.no_grad():
        out = model(rgb, audio, mask, lengths)

    print('\noutput:')
    for k, v in out.items():
        print(f'  {k:12s}: {v.shape}  min={v.float().min():.3f}  max={v.float().max():.3f}')

    embed_norms = out['embeds'].norm(dim=-1)   
    valid_norms = embed_norms[out['seg_mask']]
    print(f'\n  embed L2 norm(Valid segment, expected about 1.0): '
          f'min={valid_norms.min():.4f}  max={valid_norms.max():.4f}')

    
    print('\nprotagonist score:')
    for b in range(B):
        scores = out['final_score'][b].tolist()
        print(f'  video{b} (length={lengths[b]}): '
              f'{[f"{s:.2f}" for s in scores]}')