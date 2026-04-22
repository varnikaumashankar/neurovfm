"""
Projection Heads and MLP Modules

Various projection heads for self-supervised learning and downstream tasks,
including DINO, iBOT, contrastive learning, and general-purpose MLPs.
"""

# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from typing import List, Optional


class CSyncBatchNorm(nn.SyncBatchNorm):
    """
    Centered Synchronized Batch Normalization.
    
    Applies synchronized batch normalization across devices but only centers
    (subtracts mean) without variance normalization when with_var=False.
    
    Args:
        with_var (bool): Whether to include variance normalization. Defaults to False.
        *args, **kwargs: Arguments passed to nn.SyncBatchNorm
    """
    def __init__(self, *args, with_var=False, **kwargs):
        super(CSyncBatchNorm, self).__init__(*args, **kwargs)
        self.with_var = with_var

    def forward(self, x):
        # center norm
        self.training = False
        if not self.with_var:
            self.running_var = torch.ones_like(self.running_var)
        normed_x = super(CSyncBatchNorm, self).forward(x)
        # update center
        self.training = True
        _ = super(CSyncBatchNorm, self).forward(x)
        return normed_x

class CustomSequential(nn.Sequential):
    """
    Custom Sequential module with automatic dimension permutation for batch norm.
    
    Automatically handles dimension permutation for batch normalization layers
    when input has more than 2 dimensions.
    """
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)

    def forward(self, input):
        for module in self:
            dim = len(input.shape)
            if isinstance(module, self.bn_types) and dim > 2:
                perm = list(range(dim - 1)); perm.insert(1, dim - 1)
                inv_perm = list(range(dim)) + [1]; inv_perm.pop(1)
                input = module(input.permute(*perm)).permute(*inv_perm)
            else:
                input = module(input)
        return input

class MLP_v0(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,   
        hidden_dims: Optional[List[int]] = None,
        norm: str = None,
        act: str = "gelu",
        num_layers: int = None,
        hidden_dim: int = None,
        dropout: float = 0.0,
        weight_init: str = "xavier",
        task: str = "regression", # "regression", "classification", "dual"
        num_classes: int = 3,
    ):
        super().__init__()

        if task not in {"regression", "classification", "dual"}:
            raise ValueError(f"Unknown task '{task}'. Choose from 'regression', 'classification', or 'dual'.")

        self.task = task
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.weight_init = weight_init

        if hidden_dims is None:
            if num_layers is not None and hidden_dim is not None:
                hidden_dims = [hidden_dim] * num_layers
            else:
                hidden_dims = []

        layers = []
        prev_dim = in_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if norm:
                layers.append(self._build_norm(norm, h_dim))
            layers.append(self._build_act(act))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Shared trunk
        self.backbone = nn.Sequential(*layers) if len(layers) > 0 else nn.Identity()

        # Task-specific heads
        if self.task == "regression":
            self.regression_head = nn.Linear(prev_dim, out_dim)

        elif self.task == "classification":
            self.classification_head = nn.Linear(prev_dim, num_classes)

        elif self.task == "dual":
            self.regression_head = nn.Linear(prev_dim, out_dim)
            self.classification_head = nn.Linear(prev_dim, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.weight_init == "trunc_normal":
                trunc_normal_(m.weight, std=0.02)
            elif self.weight_init == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif self.weight_init == "kaiming":
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            else:
                raise ValueError(f"Unknown weight_init '{self.weight_init}'")

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _build_norm(self, norm, hidden_dim):
        if norm == "bn":
            return nn.BatchNorm1d(hidden_dim)
        elif norm == "syncbn":
            return nn.SyncBatchNorm(hidden_dim)
        elif norm == "csyncbn":
            return CSyncBatchNorm(hidden_dim)
        elif norm == "ln":
            return nn.LayerNorm(hidden_dim)
        else:
            raise ValueError(f"unknown norm type {norm}")

    def _build_act(self, act):
        if act == "relu":
            return nn.ReLU()
        elif act == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"unknown act type {act}")

    def forward(self, x):
        features = self.backbone(x)

        if self.task == "regression":
            return self.regression_head(features)

        elif self.task == "classification":
            return self.classification_head(features)  

        elif self.task == "dual":
            reg_out = self.regression_head(features)
            cls_out = self.classification_head(features) 
            return reg_out, cls_out

class MLP_v1(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        hidden_dims: Optional[List[int]] = None,
        norm: str = None,
        act: str = "gelu",
        num_layers: int = None,
        hidden_dim: int = None,
        dropout: float = 0.0,
        weight_init: str = "xavier",
        task: str = "regression",  # "regression", "classification", "dual"
        num_classes: int = 3,
    ):
        super().__init__()

        if task not in {"regression", "classification", "dual"}:
            raise ValueError(
                f"Unknown task '{task}'. Choose from 'regression', 'classification', or 'dual'."
            )

        self.task = task
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.weight_init = weight_init
        self.dropout = dropout
        self.act = act
        self.norm = norm

        if hidden_dims is None:
            if num_layers is not None and hidden_dim is not None:
                hidden_dims = [hidden_dim] * num_layers
            else:
                hidden_dims = []

        layers = []
        prev_dim = in_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if norm:
                layers.append(self._build_norm(norm, h_dim))
            layers.append(self._build_act(act))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers) if len(layers) > 0 else nn.Identity()

        if self.task == "regression":
            self.regression_head = nn.Linear(prev_dim, out_dim)

        elif self.task == "classification":
            self.classification_head = nn.Linear(prev_dim, num_classes)

        elif self.task == "dual":
            self.classification_head = nn.Linear(prev_dim, num_classes)

            self.regression_heads = nn.ModuleList([
                nn.Linear(prev_dim, out_dim) for _ in range(num_classes)
            ])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.weight_init == "trunc_normal":
                trunc_normal_(m.weight, std=0.02)
            elif self.weight_init == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif self.weight_init == "kaiming":
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            else:
                raise ValueError(f"Unknown weight_init '{self.weight_init}'")

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _build_norm(self, norm, hidden_dim):
        if norm == "bn":
            return nn.BatchNorm1d(hidden_dim)
        elif norm == "syncbn":
            return nn.SyncBatchNorm(hidden_dim)
        elif norm == "csyncbn":
            return CSyncBatchNorm(hidden_dim)
        elif norm == "ln":
            return nn.LayerNorm(hidden_dim)
        else:
            raise ValueError(f"unknown norm type {norm}")

    def _build_act(self, act):
        if act == "relu":
            return nn.ReLU()
        elif act == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"unknown act type {act}")

    def forward(self, x):
        features = self.backbone(x)

        if self.task == "regression":
            return self.regression_head(features)

        elif self.task == "classification":
            return self.classification_head(features)

        elif self.task == "dual":
            cls_out = self.classification_head(features)       
            cls_probs = torch.softmax(cls_out, dim=-1)           

            reg_outputs = torch.cat(
                [head(features) for head in self.regression_heads],
                dim=-1
            )                                                   

            reg_out = (cls_probs * reg_outputs).sum(dim=-1, keepdim=True)  

            return reg_out, cls_out

class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        hidden_dims: Optional[List[int]] = None,
        norm: str = None,
        act: str = "gelu",
        num_layers: int = None,
        hidden_dim: int = None,
        dropout: float = 0.0,
        weight_init: str = "xavier",
        task: str = "regression",  # "regression", "classification", "dual"
        num_classes: int = 3,
        use_softmax_for_reg: bool = True, 
    ):
        super().__init__()

        if task not in {"regression", "classification", "dual"}:
            raise ValueError(
                f"Unknown task '{task}'. Choose from 'regression', 'classification', or 'dual'."
            )

        self.task = task
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.weight_init = weight_init
        self.use_softmax_for_reg = use_softmax_for_reg
        self.norm = norm
        self.act = act
        self.dropout = dropout

        if hidden_dims is None:
            if num_layers is not None and hidden_dim is not None:
                hidden_dims = [hidden_dim] * num_layers
            else:
                hidden_dims = []

        layers = []
        prev_dim = in_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if norm:
                layers.append(self._build_norm(norm, h_dim))
            layers.append(self._build_act(act))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        
        self.backbone = nn.Sequential(*layers) if len(layers) > 0 else nn.Identity()

        if self.task == "regression":
            self.regression_head = nn.Linear(prev_dim, out_dim)

        elif self.task == "classification":
            self.classification_head = nn.Linear(prev_dim, num_classes)

        elif self.task == "dual":
            self.classification_head = nn.Linear(prev_dim, num_classes)

            reg_in_dim = prev_dim + num_classes
            self.regression_head = nn.Sequential(
                nn.Linear(reg_in_dim, prev_dim),
                self._build_act(act),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(prev_dim, out_dim),
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.weight_init == "trunc_normal":
                trunc_normal_(m.weight, std=0.02)
            elif self.weight_init == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif self.weight_init == "kaiming":
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            else:
                raise ValueError(f"Unknown weight_init '{self.weight_init}'")

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _build_norm(self, norm, hidden_dim):
        if norm == "bn":
            return nn.BatchNorm1d(hidden_dim)
        elif norm == "syncbn":
            return nn.SyncBatchNorm(hidden_dim)
        elif norm == "csyncbn":
            return CSyncBatchNorm(hidden_dim)
        elif norm == "ln":
            return nn.LayerNorm(hidden_dim)
        else:
            raise ValueError(f"unknown norm type {norm}")

    def _build_act(self, act):
        if act == "relu":
            return nn.ReLU()
        elif act == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"unknown act type {act}")

    def forward(self, x):
        features = self.backbone(x)

        if self.task == "regression":
            return self.regression_head(features)

        elif self.task == "classification":
            return self.classification_head(features)

        elif self.task == "dual":
            cls_out = self.classification_head(features)  

            if self.use_softmax_for_reg:
                cls_signal = torch.softmax(cls_out, dim=-1)
            else:
                cls_signal = cls_out

            reg_input = torch.cat([features, cls_signal], dim=-1)
            reg_out = self.regression_head(reg_input)

            return reg_out, cls_out