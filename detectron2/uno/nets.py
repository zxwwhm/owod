from optparse import Values
from detectron2.utils.store import Store
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from .sinkhorn_knopp import SinkhornKnopp


class Prototypes(nn.Module):
    def __init__(self, output_dim, num_prototypes):
        """
        num_prototypes unlabel 类别的数量
        """
        super().__init__()

        self.prototypes = nn.Linear(output_dim, num_prototypes, bias=False)

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.prototypes.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.prototypes.weight.copy_(w)

    def forward(self, x):
        return self.prototypes(x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=1):
        """
        """
        super().__init__()

        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_hidden_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MultiHead(nn.Module):
    def __init__(
            self, input_dim, hidden_dim, output_dim, num_prototypes, num_heads, num_hidden_layers=1
    ):
        super().__init__()
        self.num_heads = num_heads

        # projectors
        self.projectors = torch.nn.ModuleList(
            [MLP(input_dim, hidden_dim, output_dim, num_hidden_layers) for _ in range(num_heads)]
        )

        # prototypes
        self.prototypes = torch.nn.ModuleList(
            [Prototypes(output_dim, num_prototypes) for _ in range(num_heads)]
        )
        self.normalize_prototypes()

    @torch.no_grad()
    def normalize_prototypes(self):
        for p in self.prototypes:
            p.normalize_prototypes()

    def forward_head(self, head_idx, feats):
        """
        head_idx 指定是第几个head
        """
        z = self.projectors[head_idx](feats)
        z = F.normalize(z, dim=1)
        return self.prototypes[head_idx](z), z

    def forward(self, feats):
        out = [self.forward_head(h, feats) for h in range(self.num_heads)]
        return [torch.stack(o) for o in map(list, zip(*out))]


class MultiHeadResNet(nn.Module):
    def __init__(
            self,
            num_labeled,
            num_unlabeled,
            hidden_dim=2048,
            proj_dim=256,
            overcluster_factor=3,
            num_heads=1,
            num_hidden_layers=1,
    ):
        super().__init__()

        # self.unknown_store = Store(num_unlabeled,20)
        # todo 2048
        self.feat_dim = 2048

        # head for label
        self.head_lab = Prototypes(self.feat_dim, num_labeled)

        if num_heads is not None:
            # head for unlabel
            self.head_unlab = MultiHead(
                input_dim=self.feat_dim,
                hidden_dim=hidden_dim,
                output_dim=proj_dim,
                num_prototypes=num_unlabeled,
                num_heads=num_heads,
                num_hidden_layers=num_hidden_layers,
            )

            # head for unlabeled overcluster
            self.head_unlab_over = MultiHead(
                input_dim=self.feat_dim,
                hidden_dim=hidden_dim,
                output_dim=proj_dim,

                num_prototypes=num_unlabeled * overcluster_factor,
                num_heads=num_heads,
                num_hidden_layers=num_hidden_layers,
            )

        self.sk = SinkhornKnopp(
            num_iters=3, epsilon=0.05
        )
        self.seen_class = num_labeled

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    @torch.no_grad()
    def _reinit_all_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def normalize_prototypes(self):
        self.head_lab.normalize_prototypes()
        if getattr(self, "head_unlab", False):
            self.head_unlab.normalize_prototypes()
            self.head_unlab_over.normalize_prototypes()

    def forward_heads(self, feats):
        out = {"logits_lab": self.head_lab(F.normalize(feats))}
        if hasattr(self, "head_unlab"):
            logits_unlab, proj_feats_unlab = self.head_unlab(feats)
            logits_unlab_over, proj_feats_unlab_over = self.head_unlab_over(feats)
            out.update(
                {
                    "logits_unlab": logits_unlab,
                    "proj_feats_unlab": proj_feats_unlab,
                    "logits_unlab_over": logits_unlab_over,
                    "proj_feats_unlab_over": proj_feats_unlab_over,
                }
            )
        return out

    def forward(self, feats):
        if isinstance(feats, list):
            out = [self.forward_heads(f) for f in feats]
            out_dict = {"feats": torch.stack(feats)}
            for key in out[0].keys():
                out_dict[key] = torch.stack([o[key] for o in out])
            return out_dict
        else:
            out = self.forward_heads(feats)
            out["feats"] = feats
            return out

    def cross_entropy_loss(self, preds, targets):
        preds = F.log_softmax(preds / 0.1, dim=-1)
        return torch.mean(-torch.sum(targets * preds, dim=-1), dim=-1)

    def swapped_prediction(self, logits, targets):
        loss = 0
        for view in range(2):
            for other_view in np.delete(range(2), view):
                loss += self.cross_entropy_loss(logits[other_view], targets[view])
        return loss / 2

    def get_uno_loss(self, feats, labels, mask_lab):
        labels = labels.long()
        self.normalize_prototypes()
        outputs = self(feats)
        
        device=feats[0].device

        outputs["logits_lab"] = (
            outputs["logits_lab"].unsqueeze(1).expand(-1, 1, -1, -1)
        )
        # print(outputs["logits_lab"].shape, outputs["logits_unlab"].shape)
        logits = torch.cat([outputs["logits_lab"], outputs["logits_unlab"]], dim=-1)
        logits_over = torch.cat([outputs["logits_lab"], outputs["logits_unlab_over"]], dim=-1)
        targets_lab = (
            F.one_hot(labels[mask_lab], num_classes=self.seen_class).float().to(labels.device)
        )

        # 10*20 unknwon feature 1 targets-->5000 si 10 sj 10 sij 10 。
        targets = torch.zeros_like(logits)
        targets_over = torch.zeros_like(logits_over)
        nlc = self.seen_class

        unknwon_feature = feats[0][~mask_lab]
        unknwon_logits=logits[0,0,~mask_lab,nlc:]
        unknwon_num=unknwon_feature.shape[0]
        feature_dim=unknwon_feature.shape[1]
        unknwon_max_logits=torch.zeros((unknwon_num,unknwon_num),device=device)
        # unknown_targets=torch.zeros_like(unknwon_max_logits)
        for i in range(unknwon_num):
            for j in range(unknwon_num):
                unknwon_max_logits[i,j]=torch.max(unknwon_logits[i]*unknwon_logits[j])
        unknwon_max_logits=unknwon_max_logits.view(-1)
        data1=unknwon_feature[:,None,:].expand(-1,unknwon_num,-1).reshape(-1,feature_dim) 
        data2=unknwon_feature[None,...].expand(unknwon_num,-1,-1).reshape(-1,feature_dim)
        cos_sim=self.cos(data1,data2)
        unknwon_targets=(cos_sim>=0.8).to(torch.float32)
        loss_pair=F.binary_cross_entropy_with_logits(
            unknwon_max_logits,
            unknwon_targets,
            reduction="mean",
        )

        for v in range(2):
            for h in range(1):
                targets[v, h, mask_lab, :nlc] = targets_lab.type_as(targets)
                targets_over[v, h, mask_lab, :nlc] = targets_lab.type_as(targets)
                targets[v, h, ~mask_lab, nlc:] = self.sk(
                    outputs["logits_unlab"][v, h, ~mask_lab]
                ).type_as(targets)
                targets_over[v, h, ~mask_lab, nlc:] = self.sk(
                    outputs["logits_unlab_over"][v, h, ~mask_lab]
                ).type_as(targets)

        loss_cluster = self.swapped_prediction(logits, targets)
        loss_overcluster = self.swapped_prediction(logits_over, targets_over)
        loss_cluster = loss_cluster.mean()
        loss_overcluster = loss_overcluster.mean()
        loss = (loss_cluster + loss_overcluster) / 2
        loss+=loss_pair
        return {"uno_loss": loss}