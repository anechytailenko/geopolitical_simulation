"""The model (plans/03 §2): a heterogeneous, edge-aware spatio-temporal GNN.

Per month: a shared 2-hop heterogeneous encoder (HeteroConv wrapping TransformerConv on the
edge-feature-rich SNAPSHOT relation + GATv2Conv on the structural relations) produces a
Country embedding. The W=12 monthly Country embeddings form a per-node sequence fed to a GRU,
then attention-pooled over time. A direction-aware concat+MLP decoder maps (h_u, h_v, edge)
to 5 class logits. Identity embeddings are concatenated at input, per node type.

Forward consumes one window = list[HeteroData] (length W); the graphs are tiny (~232 nodes)
so we encode them in a Python loop and stack — the only "custom glue" the plan calls out.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv, TransformerConv

from .config import Config, NUM_CLASSES
from .dataset import C, A, REL_SNAP, REL_BORDER, REL_MEMBER, REL_RMEMBER


class SpatioTemporalEdgeClassifier(nn.Module):
    def __init__(self, num_country: int, num_actor: int, country_in: int, actor_in: int,
                 edge_dim: int, cfg: Config):
        super().__init__()
        d, h = cfg.hidden, cfg.heads
        assert d % h == 0, "hidden must be divisible by heads"
        self.cfg = cfg
        self.num_country = num_country
        self.num_actor = num_actor

        self.emb_country = nn.Embedding(num_country, cfg.id_dim)
        self.emb_actor = nn.Embedding(num_actor, cfg.id_dim)
        self.in_country = nn.Linear(country_in + cfg.id_dim, d)
        self.in_actor = nn.Linear(actor_in + cfg.id_dim, d)

        self.convs = nn.ModuleList()
        self.norm_c = nn.ModuleList()
        self.norm_a = nn.ModuleList()
        for _ in range(cfg.hops):
            conv = HeteroConv({
                REL_SNAP: TransformerConv(d, d // h, heads=h, edge_dim=edge_dim, beta=True, dropout=cfg.dropout),
                REL_BORDER: GATv2Conv(d, d // h, heads=h, add_self_loops=True, dropout=cfg.dropout),
                REL_MEMBER: GATv2Conv((d, d), d // h, heads=h, add_self_loops=False, dropout=cfg.dropout),
                REL_RMEMBER: GATv2Conv((d, d), d // h, heads=h, add_self_loops=False, dropout=cfg.dropout),
            }, aggr="sum")
            self.convs.append(conv)
            self.norm_c.append(nn.LayerNorm(d))
            self.norm_a.append(nn.LayerNorm(d))

        self.gru = nn.GRU(d, d, num_layers=cfg.gru_layers, batch_first=True)
        self.temporal_attn = nn.Linear(d, 1)
        self.dropout = nn.Dropout(cfg.dropout)
        self.decoder = nn.Sequential(
            nn.Linear(3 * d + edge_dim, d), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(d, NUM_CLASSES),
        )
        self.register_buffer("_c_ids", torch.arange(num_country), persistent=False)
        self.register_buffer("_a_ids", torch.arange(num_actor), persistent=False)

    # -- one monthly graph -> Country embeddings [Nc, d] ------------------------------
    def encode_snapshot(self, data) -> torch.Tensor:
        xc = torch.cat([data[C].x, self.emb_country(self._c_ids)], dim=-1)
        xa = torch.cat([data[A].x, self.emb_actor(self._a_ids)], dim=-1)
        x_dict = {C: F.relu(self.in_country(xc)), A: F.relu(self.in_actor(xa))}

        edge_index_dict = {
            REL_SNAP: data[REL_SNAP].edge_index,
            REL_BORDER: data[REL_BORDER].edge_index,
            REL_MEMBER: data[REL_MEMBER].edge_index,
            REL_RMEMBER: data[REL_RMEMBER].edge_index,
        }
        edge_attr_dict = {REL_SNAP: data[REL_SNAP].edge_attr}

        for conv, nc, na in zip(self.convs, self.norm_c, self.norm_a):
            out = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            # .get fallback: a month may have no edges of some relation -> keep prior state.
            oc = out.get(C, x_dict[C])
            oa = out.get(A, x_dict[A])
            hc = self.dropout(F.elu(nc(oc))) + x_dict[C]   # residual (mitigates oversmoothing)
            ha = self.dropout(F.elu(na(oa))) + x_dict[A]
            x_dict = {C: hc, A: ha}
        return x_dict[C]

    # -- full window -> 5-class logits per target pair --------------------------------
    def forward(self, window: list, pair_index: torch.Tensor, pair_attr: torch.Tensor) -> torch.Tensor:
        seq = [self.encode_snapshot(d) for d in window]      # W x [Nc, d]
        H = torch.stack(seq, dim=1)                          # [Nc, W, d]
        gru_out, _ = self.gru(H)                             # [Nc, W, d]
        attn = torch.softmax(self.temporal_attn(gru_out), dim=1)   # [Nc, W, 1]
        h = (gru_out * attn).sum(dim=1)                      # [Nc, d]

        u = h[pair_index[:, 0]]
        v = h[pair_index[:, 1]]
        feat = torch.cat([u, v, u * v, pair_attr], dim=-1)
        return self.decoder(feat)                            # [P, 5] logits

    @classmethod
    def from_dataset(cls, ds, cfg: Config) -> "SpatioTemporalEdgeClassifier":
        return cls(
            num_country=ds.num_country, num_actor=ds.num_actor,
            country_in=ds.pp.country_feat_dim, actor_in=ds.pp.actor_feat_dim,
            edge_dim=ds.pp.edge_dim, cfg=cfg,
        )
