import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

import models
from models import register
from utils import make_coord
from .CrossAttention import CrossScaleAttention

@register('liif')
class LIIF(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        multi_scale = [2]
        self.local_size = 2

        self.encoder = models.make(encoder_spec)

        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            imnet_dim = self.encoder.out_dim

            imnet_in_dim *= 9
            self.imnet_q = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
            self.imnet_k = models.make(imnet_spec, args={'in_dim': imnet_in_dim+2, 'out_dim':imnet_in_dim})  #去掉Cell,需要更改纬度为2
            self.imnet_v = models.make(imnet_spec, args={'in_dim': imnet_in_dim+2, 'out_dim':imnet_in_dim})

        else:
            self.imnet = None

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    #implicit attention中的query_rgb()
    def query_rgb(self, coord, scale=None):
        """Query RGB value of GT.
    
        Copyright (c) 2020, Yinbo Chen, under BSD 3-Clause License.
    
        Args:
            feature (Tensor): encoded feature.
            coord (Tensor): coord tensor, shape (BHW, 2).
    
        Returns:
            result (Tensor): (part of) output.
        """
        #pdb.set_trace()
        feature = self.feat
        #res_features = []
    
        B, C, H, W = feature.shape  # [16, 64, 48, 48]
    
        if self.feat_unfold:
            feat_q = F.unfold(feature, 3, padding=1).view(B, C * 9, H, W)  # [16, 576, 48, 48]
            feat_k = F.unfold(feature, 3, padding=1).view(B, C * 9, H, W)  # [16, 576, 48, 48]
            feat_v = F.unfold(feature, 3, padding=1).view(B, C * 9, H, W)  # [16, 576, 48, 48]
        else:
            feat_q = feat_k = feat_v = feature

        query = F.grid_sample(feat_q, coord.flip(-1).unsqueeze(1), mode='nearest',
                              align_corners=False).permute(0, 3, 2, 1).contiguous()  # [16, 2304, 1, 576]
    
        feat_coord = make_coord(feature.shape[-2:], flatten=False).permute(2, 0, 1) \
            .unsqueeze(0).expand(B, 2, *feature.shape[-2:])  # [16, 2, 48, 48]
        feat_coord = feat_coord.to(coord)
        #pdb.set_trace()
        if self.local_size == 1:
            v_lst = [(0, 0)]
        else:
            v_lst = [(i, j) for i in range(-1, 2, 4 - self.local_size) for j in range(-1, 2, 4 - self.local_size)]
        eps_shift = 1e-6
        preds_k, preds_v = [], []
        #pdb.set_trace()
        for v in v_lst:
            vx, vy = v[0], v[1]
            # project to LR field
            tx = ((H - 1) / (1 - scale[:, 0, 0])).view(B, 1)  # [16, 1]
            ty = ((W - 1) / (1 - scale[:, 0, 1])).view(B, 1)  # [16, 1]
            rx = (2 * abs(vx) - 1) / tx if vx != 0 else 0  # [16, 1]
            ry = (2 * abs(vy) - 1) / ty if vy != 0 else 0  # [16, 1]
    
            bs, q = coord.shape[:2]
            coord_ = coord.clone()  # [16, 2304, 2]
            if vx != 0:
                coord_[:, :, 0] += vx / abs(vx) * rx + eps_shift  # [16, 2304]
            if vy != 0:
                coord_[:, :, 1] += vy / abs(vy) * ry + eps_shift  # [16, 2304]
            coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
    
            # key and value
            key = F.grid_sample(feat_k, coord_.flip(-1).unsqueeze(1), mode='nearest',
                                align_corners=False)[:, :, 0, :].permute(0, 2, 1).contiguous()  # [16, 2304, 576]
            value = F.grid_sample(feat_v, coord_.flip(-1).unsqueeze(1), mode='nearest',
                                  align_corners=False)[:, :, 0, :].permute(0, 2, 1).contiguous()  # [16, 2304, 576]
    
            # Interpolate K to HR resolution
            coord_k = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1),
                                    mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2,
                                                                                             1)  # [16, 2304, 2]
    
            Q, K = coord, coord_k  # [16, 2304, 2]
            rel = Q - K  # [16, 2304, 2]
            rel[:, :, 0] *= feature.shape[-2]  # without mul
            rel[:, :, 1] *= feature.shape[-1]
            inp = rel  # [16, 2304, 2]
    
            scale_ = scale.clone()  # [16, 2304, 2]
            scale_[:, :, 0] *= feature.shape[-2]
            scale_[:, :, 1] *= feature.shape[-1]

            inp_v = torch.cat([value, inp], dim=-1)  # [16, 2304, 578]
            inp_k = torch.cat([key, inp], dim=-1)  # [16, 2304, 578]
    
            inp_k = inp_k.contiguous().view(bs * q, -1)
            inp_v = inp_v.contiguous().view(bs * q, -1)
            #pdb.set_trace()
            weight_k = self.imnet_k(inp_k).view(bs, q, -1).contiguous()  # [16, 2304, 576]
            pred_k = (key * weight_k).view(bs, q, -1)  # [16, 2304, 576]

            weight_v = self.imnet_v(inp_v).view(bs, q, -1).contiguous()  # [16, 2304, 576]
            pred_v = (value * weight_v).view(bs, q, -1)  # [16, 2304, 576]
    
            preds_v.append(pred_v)
            preds_k.append(pred_k)
        #pdb.set_trace()
        preds_k = torch.stack(preds_k, dim=-1)  # [16, 2304, 576, 4]
        preds_v = torch.stack(preds_v, dim=-2)  # [16, 2304, 4, 576]
    
        attn = (query @ preds_k)  # [16, 2304, 1, 4]
        x = attn.softmax(dim=-1)@preds_v # [16, 2304, 1, 576]
        x = x.view(bs * q, -1)  # [16*2304, 576]

        #pdb.set_trace()
        result=x
        result = self.imnet_q(result)  # [16, 2304, 3]
        result = result.view(bs, q, -1)
    
        return result

    def forward(self, inp, coord, cell):
        #pdb.set_trace()
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
