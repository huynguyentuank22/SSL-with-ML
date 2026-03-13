"""
ssl_models.py  (paper-faithful refactor)
=========================================
Faithful implementations of 6 SSL tabular methods.

Fidelity levels (documented per model):
  DAE    – Faithful: encoder-decoder with 3 configurable corruption modes.
  VIME   – Faithful: per-feature empirical marginal sampling, Bernoulli mask,
             BCE(mask) + alpha*MSE(values) objective. [NeurIPS 2020]
  SCARF  – Faithful: per-feature marginal corruption, NT-Xent contrastive,
             downstream extracts encoder (not projector). [ICLR 2022]
  SAINT  – Faithful: per-feature numeric tokenisation, interleaved column
             attention + row (intersample) attention blocks, contrastive
             pre-training with two corrupted views. [Somepalli et al. 2021]
  SubTab – Faithful: random-overlapping feature subsets, shared encoder,
             pairwise multi-view NT-Xent + reconstruction loss, mean-
             aggregated inference across all subsets. [NeurIPS 2021]
  TabNet – Faithful (custom): shared+step GLU feature transformer, attentive
             transformer with prior-scale update, self-supervised pre-training
             by masked-feature reconstruction. If pytorch-tabnet is installed,
             pretraining uses the official library and we attach hooks for
             per-step embedding extraction. [Arik & Pfister 2021]

Interface (unchanged throughout):
  fit_ssl(X_train)
  list_available_layers() -> List[str]
  extract_embeddings(X, layer_name) -> np.ndarray
"""

import gc
import logging
import math
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional: official pytorch-tabnet
# ---------------------------------------------------------------------------
try:
    from pytorch_tabnet.pretraining import TabNetPretrainer as _TabNetPretrainer
    _HAS_PYTORCH_TABNET = True
    log.info("pytorch-tabnet found -- TabNetModel will use official pre-trainer.")
except ImportError:
    _HAS_PYTORCH_TABNET = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================================
# Shared helpers
# ===========================================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_tensor(X: np.ndarray) -> torch.Tensor:
    return torch.tensor(X, dtype=torch.float32)


def make_loader(X: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(to_tensor(X))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ===========================================================================
# Base class
# ===========================================================================

class BaseSSLModel(ABC):
    """Abstract base for all SSL tabular wrappers."""

    name: str = "base"

    def __init__(self, input_dim: int, config: dict, seed: int = 42):
        self.input_dim = input_dim
        self.config    = config
        self.seed      = seed
        self.device    = DEVICE
        self._fitted   = False

    @abstractmethod
    def fit_ssl(self, X_train: np.ndarray) -> None: ...

    @abstractmethod
    def extract_embeddings(self, X: np.ndarray, layer_name: str) -> np.ndarray: ...

    @abstractmethod
    def list_available_layers(self) -> List[str]: ...

    def save_checkpoint(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._get_state(), path)
        log.info(f"Checkpoint saved -> {path}")

    def load_checkpoint(self, path: Path):
        state = torch.load(path, map_location=self.device)
        self._set_state(state)
        self._fitted = True
        log.info(f"Checkpoint loaded <- {path}")

    def _get_state(self): raise NotImplementedError
    def _set_state(self, state): raise NotImplementedError


# ===========================================================================
# 1. DAE -- Denoising Autoencoder
#    Faithful to the canonical DAE framework (Vincent et al. 2008/2010).
#
#    Corruption modes:
#      "gaussian" -- x~_i = x_i + eps,   eps ~ N(0, noise_std)
#      "masking"  -- zero-out p_mask fraction of features
#      "swap"     -- replace p_mask features with values from a random other
#                    row in the batch (batch-level empirical marginal sampling)
#
#    Architecture: corrupt(x) -> encoder MLP -> bottleneck -> decoder MLP -> x_hat
#    Objective:    MSE(x_hat, x_clean)
#
#    Layer names: encoder_hidden_1 ... encoder_hidden_K, bottleneck
# ===========================================================================

class _DAEEncoder(nn.Module):
    def __init__(self, input_dim: int, enc_dims: List[int], latent_dim: int):
        super().__init__()
        self.layers = nn.ModuleList()
        self.acts   = nn.ModuleList()
        in_d = input_dim
        for h in enc_dims:
            self.layers.append(nn.Linear(in_d, h))
            self.acts.append(nn.ReLU())
            in_d = h
        self.latent = nn.Linear(in_d, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lin, act in zip(self.layers, self.acts):
            x = act(lin(x))
        return F.relu(self.latent(x))

    def forward_up_to(self, x: torch.Tensor, stop: int) -> torch.Tensor:
        for i, (lin, act) in enumerate(zip(self.layers, self.acts)):
            x = act(lin(x))
            if i == stop:
                return x
        return x


class DAEModel(BaseSSLModel):
    """
    Denoising Autoencoder (DAE).

    Architecture: corrupt(x) -> encoder MLP -> bottleneck -> decoder MLP -> x_hat
    Objective:    MSE(x_hat, x_clean)

    Corruption modes
    ----------------
    gaussian : x~ = x + eps,  eps ~ N(0, noise_std)
    masking  : zero-out p_mask fraction of features
    swap     : replace p_mask features with values from a random other row
               (batch-level empirical marginal sampling)

    Layer names: encoder_hidden_1 ... encoder_hidden_K, bottleneck
    """

    name = "dae"

    ARCH_CONFIGS = {
        "dae_small":  {"enc_dims": [64],          "latent_dim": 32,
                       "noise_std": 0.2, "p_mask": 0.1,  "mode": "gaussian"},
        "dae_medium": {"enc_dims": [128, 64],      "latent_dim": 64,
                       "noise_std": 0.1, "p_mask": 0.15, "mode": "masking"},
        "dae_large":  {"enc_dims": [256, 128, 64], "latent_dim": 64,
                       "noise_std": 0.1, "p_mask": 0.15, "mode": "swap"},
    }

    def __init__(
        self,
        input_dim: int,
        arch_name: str = "dae_medium",
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        seed: int = 42,
    ):
        arch = self.ARCH_CONFIGS[arch_name]
        config = dict(arch_name=arch_name, epochs=epochs,
                      batch_size=batch_size, lr=lr, **arch)
        super().__init__(input_dim, config, seed)
        self.arch_name = arch_name

        enc_dims = arch["enc_dims"]
        latent   = arch["latent_dim"]

        self.encoder = _DAEEncoder(input_dim, enc_dims, latent).to(self.device)

        # Symmetric decoder
        dec_dims: List[nn.Module] = []
        in_d = latent
        for h in reversed(enc_dims):
            dec_dims += [nn.Linear(in_d, h), nn.ReLU()]
            in_d = h
        dec_dims.append(nn.Linear(in_d, input_dim))
        self.decoder = nn.Sequential(*dec_dims).to(self.device)

    # ------------------------------------------------------------------
    def _corrupt(self, x: torch.Tensor) -> torch.Tensor:
        mode = self.config["mode"]
        if mode == "gaussian":
            return x + self.config["noise_std"] * torch.randn_like(x)
        elif mode == "masking":
            mask = torch.rand_like(x) < self.config["p_mask"]
            return x.masked_fill(mask, 0.0)
        elif mode == "swap":
            p    = self.config["p_mask"]
            mask = torch.rand_like(x) < p
            perm = torch.randperm(x.size(0), device=x.device)
            return torch.where(mask, x[perm], x)
        else:
            raise ValueError(f"Unknown DAE corruption mode: {mode!r}")

    def fit_ssl(self, X_train: np.ndarray) -> None:
        set_seed(self.seed)
        loader = make_loader(X_train, self.config["batch_size"])
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optim  = torch.optim.Adam(params, lr=self.config["lr"])
        mse    = nn.MSELoss()

        log.info(f"[DAE-{self.arch_name}] Pretraining {self.config['epochs']} epochs ...")
        for epoch in range(1, self.config["epochs"] + 1):
            self.encoder.train(); self.decoder.train()
            total = 0.0
            for (x,) in loader:
                x       = x.to(self.device)
                x_tilde = self._corrupt(x)
                z       = self.encoder(x_tilde)
                x_hat   = self.decoder(z)
                loss    = mse(x_hat, x)
                optim.zero_grad(); loss.backward(); optim.step()
                total += loss.item()
            if epoch % max(1, self.config["epochs"] // 5) == 0 or epoch == 1:
                log.info(f"  epoch {epoch:3d}/{self.config['epochs']}  loss={total/len(loader):.4f}")
        self._fitted = True

    def list_available_layers(self) -> List[str]:
        n = len(self.encoder.layers)
        return [f"encoder_hidden_{i+1}" for i in range(n)] + ["bottleneck"]

    @torch.no_grad()
    def extract_embeddings(self, X: np.ndarray, layer_name: str) -> np.ndarray:
        self.encoder.eval()
        out_list = []
        for (x,) in make_loader(X, batch_size=512, shuffle=False):
            x = x.to(self.device)
            if layer_name == "bottleneck":
                out = self.encoder(x)
            else:
                idx = int(layer_name.split("_")[-1]) - 1
                out = self.encoder.forward_up_to(x, idx)
            out_list.append(out.cpu().numpy())
        return np.concatenate(out_list, axis=0)

    def _get_state(self):
        return {"encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "config":  self.config}

    def _set_state(self, state):
        self.encoder.load_state_dict(state["encoder"])
        self.decoder.load_state_dict(state["decoder"])


# ===========================================================================
# 2. VIME -- Value Imputation and Mask Estimation  [NeurIPS 2020]
#    Faithful to: Yoon et al. NeurIPS 2020.
#
#    Corruption (Section 3.1):
#      m_{ij}  ~ Bernoulli(p_m)
#      x_bar_{ij} = X_ref[k_{ij}, j],  k_{ij} ~ Uniform(|X_ref|)
#      x_tilde_{ij} = m_{ij} * x_bar_{ij} + (1 - m_{ij}) * x_{ij}
#
#    Objective:
#      L = L_BCE(m_hat, m)  +  alpha * L_MSE(x_hat * m, x * m)
#
#    Layer names: hidden_1 ... hidden_K, latent
# ===========================================================================

class _VIMEEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        self.layers = nn.ModuleList()
        self.acts   = nn.ModuleList()
        in_d = input_dim
        for h in hidden_dims:
            self.layers.append(nn.Linear(in_d, h))
            self.acts.append(nn.ReLU())
            in_d = h
        self.latent = nn.Linear(in_d, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lin, act in zip(self.layers, self.acts):
            x = act(lin(x))
        return self.latent(x)

    def forward_up_to(self, x: torch.Tensor, stop: int) -> torch.Tensor:
        for i, (lin, act) in enumerate(zip(self.layers, self.acts)):
            x = act(lin(x))
            if i == stop:
                return x
        return x



class VIMEModel(BaseSSLModel):
    """
    VIME: Value Imputation and Mask Estimation (Yoon et al. NeurIPS 2020).

    Corruption uses per-feature empirical marginal sampling: masked feature j
    in sample i is replaced by X_ref[k, j] where k is drawn uniformly from
    the training reference buffer (up to 50 000 rows for memory efficiency).

    Objective: L = BCE(m_hat, m) + alpha * MSE(x_hat * m, x * m)

    Layer names: hidden_1 ... hidden_K, latent
    """

    name = "vime"

    ARCH_CONFIGS = {
        "vime_small":  {"hidden_dims": [64],       "latent_dim": 32},
        "vime_medium": {"hidden_dims": [128, 64],   "latent_dim": 64},
        "vime_large":  {"hidden_dims": [256, 128],  "latent_dim": 128},
    }

    def __init__(
        self,
        input_dim: int,
        arch_name: str = "vime_medium",
        p_mask: float  = 0.3,
        alpha: float   = 2.0,
        epochs: int    = 50,
        batch_size: int = 256,
        lr: float      = 1e-3,
        seed: int      = 42,
    ):
        arch = self.ARCH_CONFIGS[arch_name]
        config = dict(arch_name=arch_name, p_mask=p_mask, alpha=alpha,
                      epochs=epochs, batch_size=batch_size, lr=lr, **arch)
        super().__init__(input_dim, config, seed)
        self.arch_name = arch_name

        self.encoder    = _VIMEEncoder(input_dim,
                                       arch["hidden_dims"],
                                       arch["latent_dim"]).to(self.device)
        self.mask_head  = nn.Sequential(
            nn.Linear(arch["latent_dim"], input_dim), nn.Sigmoid()
        ).to(self.device)
        self.value_head = nn.Linear(arch["latent_dim"], input_dim).to(self.device)

        self._X_ref: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def _corrupt(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        VIME-faithful corruption:
          m_{ij} ~ Bernoulli(p_m)
          x_bar_{ij} = X_ref[k_{ij}, j],  k_{ij} ~ Uniform(N_ref)
          x_tilde = m * x_bar + (1-m) * x
        """
        p    = self.config["p_mask"]
        B, D = x.shape
        m    = torch.bernoulli(torch.full((B, D), p, device=x.device))

        N       = self._X_ref.size(0)
        row_idx = torch.randint(0, N, (B * D,))
        col_idx = torch.arange(D).repeat(B)
        x_bar   = self._X_ref[row_idx, col_idx].view(B, D).to(x.device)

        x_tilde = m * x_bar + (1.0 - m) * x
        return x_tilde, m

    def fit_ssl(self, X_train: np.ndarray) -> None:
        set_seed(self.seed)

        N   = len(X_train)
        ref = min(50_000, N)
        idx = np.random.choice(N, ref, replace=False)
        self._X_ref = torch.tensor(X_train[idx], dtype=torch.float32)

        loader = make_loader(X_train, self.config["batch_size"])
        params = (list(self.encoder.parameters()) +
                  list(self.mask_head.parameters()) +
                  list(self.value_head.parameters()))
        optim  = torch.optim.Adam(params, lr=self.config["lr"])
        bce    = nn.BCELoss()
        mse    = nn.MSELoss()
        alpha  = self.config["alpha"]

        log.info(f"[VIME-{self.arch_name}] Pretraining {self.config['epochs']} epochs ...")
        for epoch in range(1, self.config["epochs"] + 1):
            self.encoder.train(); self.mask_head.train(); self.value_head.train()
            total = 0.0
            for (x,) in loader:
                x = x.to(self.device)
                x_tilde, m = self._corrupt(x)

                z     = self.encoder(x_tilde)
                m_hat = self.mask_head(z)
                x_hat = self.value_head(z)

                loss_m = bce(m_hat, m)
                loss_v = (mse(x_hat * m, x * m)
                          if m.sum() > 0 else x_hat.new_zeros(1).mean())
                loss   = loss_m + alpha * loss_v

                optim.zero_grad(); loss.backward(); optim.step()
                total += loss.item()
            if epoch % max(1, self.config["epochs"] // 5) == 0 or epoch == 1:
                log.info(f"  epoch {epoch:3d}/{self.config['epochs']}  loss={total/len(loader):.4f}")
        self._fitted = True

    def list_available_layers(self) -> List[str]:
        n = len(self.encoder.layers)
        return [f"hidden_{i+1}" for i in range(n)] + ["latent"]

    @torch.no_grad()
    def extract_embeddings(self, X: np.ndarray, layer_name: str) -> np.ndarray:
        self.encoder.eval()
        out_list = []
        for (x,) in make_loader(X, batch_size=512, shuffle=False):
            x = x.to(self.device)
            if layer_name == "latent":
                out = self.encoder(x)
            else:
                idx = int(layer_name.split("_")[1]) - 1
                out = self.encoder.forward_up_to(x, idx)
            out_list.append(out.cpu().numpy())
        return np.concatenate(out_list, axis=0)

    def _get_state(self):
        return {"encoder":    self.encoder.state_dict(),
                "mask_head":  self.mask_head.state_dict(),
                "value_head": self.value_head.state_dict(),
                "X_ref":      self._X_ref,
                "config":     self.config}

    def _set_state(self, state):
        self.encoder.load_state_dict(state["encoder"])
        self.mask_head.load_state_dict(state["mask_head"])
        self.value_head.load_state_dict(state["value_head"])
        self._X_ref = state.get("X_ref")



# ===========================================================================
# 3. SCARF -- Self-Supervised Contrastive Learning using Random Feature Corruption
#    Faithful to: Bahri et al. ICLR 2022.
#
#    Augmentation: for each feature j, with probability corruption_rate,
#    replace x_{ij} with X_ref[k_j, j],  k_j ~ Uniform(N_ref).
#
#    Two views: (clean x, corrupted x~)
#    Contrastive: NT-Xent on proj_head(encoder(x)), proj_head(encoder(x~))
#    Downstream: encoder output ONLY (projection head is discarded post-training)
#
#    Layer names: hidden_1 ... hidden_K, embedding
# ===========================================================================

class _SCARFEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        self.blocks = nn.ModuleList()
        in_d = input_dim
        for h in hidden_dims:
            self.blocks.append(nn.Sequential(nn.Linear(in_d, h), nn.ReLU()))
            in_d = h
        self.proj_layer = nn.Linear(in_d, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.proj_layer(x)

    def forward_up_to(self, x: torch.Tensor, stop: int) -> torch.Tensor:
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == stop:
                return x
        return x


class SCARFModel(BaseSSLModel):
    """
    SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption
    (Bahri et al. ICLR 2022).

    Augmentation: each feature j is replaced independently with probability
    corruption_rate by X_ref[k, j], k ~ Uniform(N_ref).  This is exact
    per-feature marginal sampling as described in the paper.

    Objective: NT-Xent between proj(encoder(x_clean)) and proj(encoder(x_tilde)).
    Downstream: encoder output (projection head discarded after pretraining).

    Layer names: hidden_1 ... hidden_K, embedding
    """

    name = "scarf"

    ARCH_CONFIGS = {
        "scarf_small":  {"hidden_dims": [64],       "output_dim": 32},
        "scarf_medium": {"hidden_dims": [128, 64],   "output_dim": 64},
        "scarf_large":  {"hidden_dims": [256, 128],  "output_dim": 128},
    }

    def __init__(
        self,
        input_dim: int,
        arch_name: str         = "scarf_medium",
        corruption_rate: float = 0.6,
        temperature: float     = 0.07,
        epochs: int            = 50,
        batch_size: int        = 256,
        lr: float              = 1e-3,
        seed: int              = 42,
    ):
        arch = self.ARCH_CONFIGS[arch_name]
        config = dict(arch_name=arch_name, corruption_rate=corruption_rate,
                      temperature=temperature, epochs=epochs,
                      batch_size=batch_size, lr=lr, **arch)
        super().__init__(input_dim, config, seed)
        self.arch_name = arch_name

        self.encoder   = _SCARFEncoder(input_dim,
                                       arch["hidden_dims"],
                                       arch["output_dim"]).to(self.device)
        self.proj_head = nn.Sequential(
            nn.Linear(arch["output_dim"], arch["output_dim"]),
            nn.ReLU(),
            nn.Linear(arch["output_dim"], arch["output_dim"]),
        ).to(self.device)

        self._X_ref: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def _corrupt(self, x: torch.Tensor) -> torch.Tensor:
        p    = self.config["corruption_rate"]
        B, D = x.shape
        mask = torch.rand(B, D, device=x.device) < p

        N       = self._X_ref.size(0)
        row_idx = torch.randint(0, N, (B * D,))
        col_idx = torch.arange(D).repeat(B)
        x_bar   = self._X_ref[row_idx, col_idx].view(B, D).to(x.device)

        return torch.where(mask, x_bar, x)

    def _nt_xent(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B   = z1.size(0)
        z   = F.normalize(torch.cat([z1, z2], dim=0), dim=-1)
        sim = torch.mm(z, z.T) / self.config["temperature"]
        eye = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(eye, float("-inf"))
        labels = torch.cat([torch.arange(B, device=z.device) + B,
                             torch.arange(B, device=z.device)])
        return F.cross_entropy(sim, labels)

    def fit_ssl(self, X_train: np.ndarray) -> None:
        set_seed(self.seed)
        N   = len(X_train)
        ref = min(50_000, N)
        self._X_ref = torch.tensor(
            X_train[np.random.choice(N, ref, replace=False)], dtype=torch.float32
        )

        loader = make_loader(X_train, self.config["batch_size"])
        params = list(self.encoder.parameters()) + list(self.proj_head.parameters())
        optim  = torch.optim.Adam(params, lr=self.config["lr"])

        log.info(f"[SCARF-{self.arch_name}] Pretraining {self.config['epochs']} epochs ...")
        for epoch in range(1, self.config["epochs"] + 1):
            self.encoder.train(); self.proj_head.train()
            total = 0.0
            for (x,) in loader:
                x       = x.to(self.device)
                x_tilde = self._corrupt(x)
                z1 = self.proj_head(self.encoder(x))
                z2 = self.proj_head(self.encoder(x_tilde))
                loss = self._nt_xent(z1, z2)
                optim.zero_grad(); loss.backward(); optim.step()
                total += loss.item()
            if epoch % max(1, self.config["epochs"] // 5) == 0 or epoch == 1:
                log.info(f"  epoch {epoch:3d}/{self.config['epochs']}  loss={total/len(loader):.4f}")
        self._fitted = True

    def list_available_layers(self) -> List[str]:
        n = len(self.encoder.blocks)
        return [f"hidden_{i+1}" for i in range(n)] + ["embedding"]

    @torch.no_grad()
    def extract_embeddings(self, X: np.ndarray, layer_name: str) -> np.ndarray:
        self.encoder.eval()
        out_list = []
        for (x,) in make_loader(X, batch_size=512, shuffle=False):
            x = x.to(self.device)
            if layer_name == "embedding":
                out = self.encoder(x)
            else:
                idx = int(layer_name.split("_")[1]) - 1
                out = self.encoder.forward_up_to(x, idx)
            out_list.append(out.cpu().numpy())
        return np.concatenate(out_list, axis=0)

    def _get_state(self):
        return {"encoder":   self.encoder.state_dict(),
                "proj_head": self.proj_head.state_dict(),
                "X_ref":     self._X_ref,
                "config":    self.config}

    def _set_state(self, state):
        self.encoder.load_state_dict(state["encoder"])
        self.proj_head.load_state_dict(state["proj_head"])
        self._X_ref = state.get("X_ref")



# ===========================================================================
# 4. SAINT -- Self-Attention and Intersample Attention Transformer
#    Faithful to: Somepalli et al. 2021 (arXiv:2106.01342).
#
#    Tokenisation:
#      Each numeric feature i -> Linear_i(x_i) -> R^{d_model}
#      Implemented as weight matrix W in R^{F x d_model}.
#
#    Architecture: L blocks, each:
#      (1) Pre-LN -> column MHA (F feature tokens attend to each other) -> residual
#      (2) Pre-LN -> FFN -> residual
#      (3) Pre-LN -> row MHA (intersample: permute (B,F,d) -> (F,B,d),
#                             attend over B, permute back) -> residual
#      (4) Pre-LN -> FFN -> residual
#    Pooling: mean over F feature tokens -> (B, d_model)
#
#    SSL pretraining (contrastive + denoising):
#      L = NT-Xent(proj(enc(view_clean)), proj(enc(view_corrupt)))
#        + lambda_recon * MSE(recon_head(enc(view_corrupt)), x_clean)
#
#    NOTE: row attention is O(B^2 * F). Recommended batch_size <= 64.
#
#    Layer names: block_1 ... block_L, final_pooled
# ===========================================================================

class _SAINTNumericTokeniser(nn.Module):
    """
    Per-feature linear projection: x_i (scalar) -> R^{d_model}.
    Weight W[i] is the projection vector for feature i.
    """

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias   = nn.Parameter(torch.zeros(n_features, d_model))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)  ->  out: (B, F, d_model)
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class _IntersampleAttention(nn.Module):
    """
    Row (intersample) attention as in SAINT:
      For each feature position i, self-attention over the B batch samples.
      Input/output: (B, F, d_model). Memory: O(F * B^2).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, F, d = x.shape
        xt    = x.permute(1, 0, 2)        # (F, B, d)
        a, _  = self.attn(xt, xt, xt)
        xt    = self.norm(xt + a)
        return xt.permute(1, 0, 2)        # (B, F, d)


class _SAINTBlock(nn.Module):
    """One SAINT block: column_attn + FFN -> row_attn + FFN (all pre-LayerNorm)."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm_col = nn.LayerNorm(d_model)
        self.col_attn = nn.MultiheadAttention(d_model, n_heads,
                                              batch_first=True, dropout=dropout)
        self.norm_ff1 = nn.LayerNorm(d_model)
        self.ff1      = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm_row = nn.LayerNorm(d_model)
        self.row_attn = _IntersampleAttention(d_model, n_heads, dropout)
        self.norm_ff2 = nn.LayerNorm(d_model)
        self.ff2      = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h    = self.norm_col(x)
        a, _ = self.col_attn(h, h, h)
        x    = x + a
        x    = x + self.ff1(self.norm_ff1(x))
        x    = self.norm_row(x)
        x    = self.row_attn(x)
        x    = x + self.ff2(self.norm_ff2(x))
        return x


class _SAINTEncoder(nn.Module):
    def __init__(self, n_features: int, d_model: int, n_heads: int,
                 n_layers: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.tokeniser  = _SAINTNumericTokeniser(n_features, d_model)
        self.blocks     = nn.ModuleList([
            _SAINTBlock(d_model, n_heads, ff_dim, dropout) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.tokeniser(x)
        for blk in self.blocks:
            t = blk(t)
        return self.final_norm(t.mean(dim=1))

    def forward_up_to_block(self, x: torch.Tensor, stop: int) -> torch.Tensor:
        t = self.tokeniser(x)
        for i, blk in enumerate(self.blocks):
            t = blk(t)
            if i == stop:
                return self.final_norm(t.mean(dim=1))
        return self.final_norm(t.mean(dim=1))


class SAINTModel(BaseSSLModel):
    """
    SAINT: Self-Attention and Intersample Attention Transformer
    (Somepalli et al. arXiv:2106.01342, 2021).

    Architecture: per-feature numeric tokenisation -> L blocks, each with
    column MHA + FFN -> row (intersample) MHA + FFN (all pre-LayerNorm) ->
    mean-pool -> (B, d_model).

    SSL pretraining: contrastive (NT-Xent on two corrupted views) +
    denoising (MSE reconstruction of clean features from corrupted encoding).

    NOTE: Row attention is O(B^2 * F); use batch_size <= 64 for F=58 on 16 GB GPU.

    Layer names: block_1 ... block_L, final_pooled
    """

    name = "saint"

    ARCH_CONFIGS = {
        "saint_small":  {"d_model": 32,  "n_heads": 2, "n_layers": 2, "ff_dim": 64,  "dropout": 0.1},
        "saint_medium": {"d_model": 64,  "n_heads": 4, "n_layers": 3, "ff_dim": 128, "dropout": 0.1},
        "saint_large":  {"d_model": 128, "n_heads": 4, "n_layers": 4, "ff_dim": 256, "dropout": 0.1},
    }

    def __init__(
        self,
        input_dim: int,
        arch_name: str           = "saint_medium",
        feat_corrupt_rate: float = 0.3,
        noise_std: float         = 0.05,
        temperature: float       = 0.07,
        lambda_recon: float      = 1.0,
        epochs: int              = 50,
        batch_size: int          = 64,
        lr: float                = 1e-3,
        seed: int                = 42,
    ):
        arch = self.ARCH_CONFIGS[arch_name]
        config = dict(arch_name=arch_name, feat_corrupt_rate=feat_corrupt_rate,
                      noise_std=noise_std, temperature=temperature,
                      lambda_recon=lambda_recon, epochs=epochs,
                      batch_size=batch_size, lr=lr, **arch)
        super().__init__(input_dim, config, seed)
        self.arch_name = arch_name

        d = arch["d_model"]
        self.encoder    = _SAINTEncoder(
            input_dim, d, arch["n_heads"],
            arch["n_layers"], arch["ff_dim"], arch["dropout"],
        ).to(self.device)
        self.proj_head  = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)
        ).to(self.device)
        self.recon_head = nn.Linear(d, input_dim).to(self.device)

        if batch_size > 128:
            log.warning(
                f"[SAINT] batch_size={batch_size} may OOM due to O(B^2*F) "
                "intersample attention. Recommend <= 64."
            )

    # ------------------------------------------------------------------
    def _corrupt(self, x: torch.Tensor) -> torch.Tensor:
        p    = self.config["feat_corrupt_rate"]
        B, D = x.shape
        mask = torch.rand(B, D, device=x.device) < p
        perm = torch.randperm(B, device=x.device)
        x_cor = torch.where(mask, x[perm], x)
        return x_cor + self.config["noise_std"] * torch.randn_like(x_cor)

    def _nt_xent(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B   = z1.size(0)
        z   = F.normalize(torch.cat([z1, z2], dim=0), dim=-1)
        sim = torch.mm(z, z.T) / self.config["temperature"]
        eye = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(eye, float("-inf"))
        labels = torch.cat([torch.arange(B, device=z.device) + B,
                             torch.arange(B, device=z.device)])
        return F.cross_entropy(sim, labels)

    def fit_ssl(self, X_train: np.ndarray) -> None:
        set_seed(self.seed)
        loader = make_loader(X_train, self.config["batch_size"])
        params = (list(self.encoder.parameters()) +
                  list(self.proj_head.parameters()) +
                  list(self.recon_head.parameters()))
        optim  = torch.optim.Adam(params, lr=self.config["lr"])
        mse    = nn.MSELoss()
        lam    = self.config["lambda_recon"]

        log.info(f"[SAINT-{self.arch_name}] Pretraining {self.config['epochs']} epochs "
                 f"(batch={self.config['batch_size']}, row-attn=ON) ...")
        for epoch in range(1, self.config["epochs"] + 1):
            self.encoder.train(); self.proj_head.train(); self.recon_head.train()
            total = 0.0
            for (x,) in loader:
                x     = x.to(self.device)
                x_cor = self._corrupt(x)

                z_cl  = self.encoder(x)
                z_co  = self.encoder(x_cor)

                loss_ctr = self._nt_xent(self.proj_head(z_cl), self.proj_head(z_co))
                loss_rec = mse(self.recon_head(z_co), x)
                loss     = loss_ctr + lam * loss_rec

                optim.zero_grad(); loss.backward(); optim.step()
                total += loss.item()
            if epoch % max(1, self.config["epochs"] // 5) == 0 or epoch == 1:
                log.info(f"  epoch {epoch:3d}/{self.config['epochs']}  loss={total/len(loader):.4f}")
        self._fitted = True

    def list_available_layers(self) -> List[str]:
        n = self.config["n_layers"]
        return [f"block_{i+1}" for i in range(n)] + ["final_pooled"]

    @torch.no_grad()
    def extract_embeddings(self, X: np.ndarray, layer_name: str) -> np.ndarray:
        self.encoder.eval()
        out_list = []
        bs = min(self.config["batch_size"], 128)
        for (x,) in make_loader(X, batch_size=bs, shuffle=False):
            x = x.to(self.device)
            if layer_name == "final_pooled":
                out = self.encoder(x)
            else:
                idx = int(layer_name.split("_")[1]) - 1
                out = self.encoder.forward_up_to_block(x, idx)
            out_list.append(out.cpu().numpy())
        return np.concatenate(out_list, axis=0)

    def _get_state(self):
        return {"encoder":    self.encoder.state_dict(),
                "proj_head":  self.proj_head.state_dict(),
                "recon_head": self.recon_head.state_dict(),
                "config":     self.config}

    def _set_state(self, state):
        self.encoder.load_state_dict(state["encoder"])
        self.proj_head.load_state_dict(state["proj_head"])
        self.recon_head.load_state_dict(state["recon_head"])



# ===========================================================================
# 5. SubTab -- Subsetting Features of Tabular Data for SSL  [NeurIPS 2021]
#    Faithful to: Ucar et al. NeurIPS 2021.
#
#    Subset construction (Section 3.1):
#      1. Shuffle all F feature indices once (seeded).
#      2. Divide shuffled order into K equal base partitions (size ~F/K).
#      3. Add overlap: each partition gets floor(overlap * base_size) extra
#         features sampled randomly from its complement.
#      4. Zero-pad all subsets to max_subset_size for batch-friendly encoding.
#
#    Training (Section 3.2):
#      L = sum_{i<j} NT-Xent(p_i, p_j) / C(K,2)   [pairwise contrastive]
#        + lambda_rec * sum_k MSE(dec(z_k), x_k) / K  [reconstruction]
#
#    Inference (Section 3.3):
#      z = mean( enc(x_1), ..., enc(x_K) )   -- aggregate ALL K subsets
#
#    Layer names: aggregated_hidden_1 ... aggregated_hidden_M, aggregated_latent
# ===========================================================================

class _SubTabEncoder(nn.Module):
    def __init__(self, sub_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        self.blocks = nn.ModuleList()
        in_d = sub_dim
        for h in hidden_dims:
            self.blocks.append(nn.Sequential(nn.Linear(in_d, h), nn.ReLU()))
            in_d = h
        self.latent = nn.Linear(in_d, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.latent(x)

    def forward_up_to(self, x: torch.Tensor, stop: int) -> torch.Tensor:
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == stop:
                return x
        return x


class SubTabModel(BaseSSLModel):
    """
    SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation
    Learning (Ucar et al. NeurIPS 2021).

    Subset generation: features are randomly shuffled (seeded), divided into K
    base partitions, augmented with overlapping features from their complement,
    then zero-padded to a common encoder input size.

    Training: shared encoder produces K latent vectors; pairwise NT-Xent
    across all C(K,2) pairs + MSE reconstruction of each padded subset.

    Inference: mean of all K latent vectors (true multi-view aggregation).

    Layer names: aggregated_hidden_1 ... aggregated_hidden_M, aggregated_latent
    """

    name = "subtab"

    ARCH_CONFIGS = {
        "subtab_small":  {"hidden_dims": [64],       "latent_dim": 32,  "n_subsets": 3, "overlap": 0.5},
        "subtab_medium": {"hidden_dims": [128, 64],   "latent_dim": 64,  "n_subsets": 4, "overlap": 0.5},
        "subtab_large":  {"hidden_dims": [256, 128],  "latent_dim": 64,  "n_subsets": 4, "overlap": 0.5},
    }

    def __init__(
        self,
        input_dim: int,
        arch_name: str     = "subtab_medium",
        epochs: int        = 50,
        batch_size: int    = 256,
        lr: float          = 1e-3,
        temperature: float = 0.07,
        lambda_rec: float  = 1.0,
        seed: int          = 42,
    ):
        arch = self.ARCH_CONFIGS[arch_name]
        config = dict(arch_name=arch_name, epochs=epochs, batch_size=batch_size,
                      lr=lr, temperature=temperature, lambda_rec=lambda_rec, **arch)
        super().__init__(input_dim, config, seed)
        self.arch_name = arch_name

        rng          = np.random.default_rng(seed)
        feat_order   = rng.permutation(input_dim).tolist()
        n_subsets    = arch["n_subsets"]
        base_size    = math.ceil(input_dim / n_subsets)
        overlap_size = max(1, int(arch["overlap"] * base_size))

        base_parts = []
        for k in range(n_subsets):
            s = k * base_size
            e = min(s + base_size, input_dim)
            base_parts.append(set(feat_order[s:e]))

        all_feats = set(feat_order)
        subsets: List[List[int]] = []
        for bp in base_parts:
            complement = sorted(all_feats - bp)
            n_extra    = min(overlap_size, len(complement))
            extra      = rng.choice(complement, n_extra, replace=False).tolist()
            subsets.append(sorted(bp | set(extra)))

        self.subsets: List[List[int]] = subsets
        self.max_sub: int             = max(len(s) for s in subsets)

        self.encoder    = _SubTabEncoder(self.max_sub,
                                         arch["hidden_dims"],
                                         arch["latent_dim"]).to(self.device)
        self.proj_head  = nn.Sequential(
            nn.Linear(arch["latent_dim"], arch["latent_dim"]),
            nn.ReLU(),
            nn.Linear(arch["latent_dim"], arch["latent_dim"]),
        ).to(self.device)
        self.recon_head = nn.Linear(arch["latent_dim"], self.max_sub).to(self.device)

    # ------------------------------------------------------------------
    def _get_subset_input(self, x: torch.Tensor, subset: List[int]) -> torch.Tensor:
        xs = x[:, subset]
        if xs.size(1) < self.max_sub:
            pad = torch.zeros(xs.size(0), self.max_sub - xs.size(1), device=x.device)
            xs  = torch.cat([xs, pad], dim=1)
        return xs

    def _nt_xent(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B   = z1.size(0)
        z   = F.normalize(torch.cat([z1, z2], dim=0), dim=-1)
        sim = torch.mm(z, z.T) / self.config["temperature"]
        eye = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(eye, float("-inf"))
        labels = torch.cat([torch.arange(B, device=z.device) + B,
                             torch.arange(B, device=z.device)])
        return F.cross_entropy(sim, labels)

    def fit_ssl(self, X_train: np.ndarray) -> None:
        set_seed(self.seed)
        loader  = make_loader(X_train, self.config["batch_size"])
        params  = (list(self.encoder.parameters()) +
                   list(self.proj_head.parameters()) +
                   list(self.recon_head.parameters()))
        optim   = torch.optim.Adam(params, lr=self.config["lr"])
        mse     = nn.MSELoss()
        K       = len(self.subsets)
        lam     = self.config["lambda_rec"]

        log.info(f"[SubTab-{self.arch_name}] Pretraining {self.config['epochs']} epochs "
                 f"({K} subsets, max_sub={self.max_sub}) ...")
        for epoch in range(1, self.config["epochs"] + 1):
            self.encoder.train(); self.proj_head.train(); self.recon_head.train()
            total = 0.0
            for (x,) in loader:
                x  = x.to(self.device)
                xs = [self._get_subset_input(x, s) for s in self.subsets]
                zs = [self.encoder(xi)             for xi in xs]
                ps = [self.proj_head(zi)           for zi in zs]

                loss_ctr = x.new_zeros(1).squeeze()
                n_pairs  = 0
                for i in range(K):
                    for j in range(i + 1, K):
                        loss_ctr = loss_ctr + self._nt_xent(ps[i], ps[j])
                        n_pairs += 1
                loss_ctr = loss_ctr / max(n_pairs, 1)

                loss_rec = x.new_zeros(1).squeeze()
                for zi, xi in zip(zs, xs):
                    loss_rec = loss_rec + mse(self.recon_head(zi), xi)
                loss_rec = loss_rec / K

                loss = loss_ctr + lam * loss_rec
                optim.zero_grad(); loss.backward(); optim.step()
                total += loss.item()
            if epoch % max(1, self.config["epochs"] // 5) == 0 or epoch == 1:
                log.info(f"  epoch {epoch:3d}/{self.config['epochs']}  loss={total/len(loader):.4f}")
        self._fitted = True

    def list_available_layers(self) -> List[str]:
        n = len(self.encoder.blocks)
        return [f"aggregated_hidden_{i+1}" for i in range(n)] + ["aggregated_latent"]

    @torch.no_grad()
    def extract_embeddings(self, X: np.ndarray, layer_name: str) -> np.ndarray:
        """Aggregate representations from ALL K subsets (paper inference procedure)."""
        self.encoder.eval()
        out_list = []
        for (x,) in make_loader(X, batch_size=512, shuffle=False):
            x = x.to(self.device)
            if layer_name == "aggregated_latent":
                zs  = [self.encoder(self._get_subset_input(x, s)) for s in self.subsets]
                out = torch.stack(zs, dim=0).mean(dim=0)
            else:
                idx = int(layer_name.split("_hidden_")[1]) - 1
                hs  = [self.encoder.forward_up_to(self._get_subset_input(x, s), idx)
                       for s in self.subsets]
                out = torch.stack(hs, dim=0).mean(dim=0)
            out_list.append(out.cpu().numpy())
        return np.concatenate(out_list, axis=0)

    def _get_state(self):
        return {"encoder":    self.encoder.state_dict(),
                "proj_head":  self.proj_head.state_dict(),
                "recon_head": self.recon_head.state_dict(),
                "subsets":    self.subsets,
                "max_sub":    self.max_sub,
                "config":     self.config}

    def _set_state(self, state):
        self.encoder.load_state_dict(state["encoder"])
        self.proj_head.load_state_dict(state["proj_head"])
        self.recon_head.load_state_dict(state["recon_head"])
        self.subsets = state["subsets"]
        self.max_sub = state["max_sub"]



# ===========================================================================
# 6. TabNet -- Self-supervised pretraining
#    Faithful to: Arik & Pfister 2021.
#
#    Architecture:
#      BN(x)
#      initial_ft = FeatTransformer(BN(x))   [shared + step1-specific GLU]
#      For step k = 1..N:
#        M_k  = AttentiveTransformer(h_{k-1}) * prior_k
#        prior_{k+1} = prior_k * (gamma - M_k)
#        h_k  = FeatTransformer(M_k * BN(x))
#        step_out_k = ReLU(h_k)
#      final_latent = mean(step_out_1, ..., step_out_N)
#
#    FeatTransformer:
#      shared_GLU(x) = GLU( BN( W_shared * x ) )  in R^{step_dim}
#      step_GLU(h)   = GLU( BN( W_step   * h ) )  in R^{step_dim}
#      output = sqrt(0.5) * (shared_GLU(x) + step_GLU(shared_GLU(x)))
#
#    Pretraining: mask p_mask fraction of features, reconstruct via decoder.
#
#    Optional: if pytorch-tabnet is installed, uses official TabNetPretrainer.
#
#    Layer names: step_1 ... step_N, final_latent
# ===========================================================================

class _GLUBlock(nn.Module):
    """Shared or step-specific GLU block: BN -> FC(in -> out*2) -> GLU -> R^out."""

    def __init__(self, in_dim: int, out_dim: int,
                 shared_fc: Optional[nn.Linear] = None):
        super().__init__()
        self.fc = shared_fc if shared_fc is not None \
                  else nn.Linear(in_dim, out_dim * 2, bias=False)
        self.bn = nn.BatchNorm1d(out_dim * 2, momentum=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.glu(self.bn(self.fc(x)), dim=-1)


class _TabNetFeatTransformer(nn.Module):
    """
    Shared + step-specific GLU Feature Transformer.
    output = sqrt(0.5) * ( shared_output + step_GLU(shared_output) )
    """

    def __init__(self, in_dim: int, step_dim: int, shared_fc: nn.Linear):
        super().__init__()
        self.shared_blk = _GLUBlock(in_dim,    step_dim, shared_fc)
        self.step_blk   = _GLUBlock(step_dim, step_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_s  = self.shared_blk(x)
        h_st = self.step_blk(h_s)
        return math.sqrt(0.5) * (h_s + h_st)


class _TabNetAttentiveTransformer(nn.Module):
    """
    Attentive Transformer:
      a = BN( W * h_{k-1} ) * prior_k
      mask = softmax(a)    (softmax approximates sparsemax)
    """

    def __init__(self, step_dim: int, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(step_dim, input_dim, bias=False)
        self.bn = nn.BatchNorm1d(input_dim, momentum=0.02)

    def forward(self, h: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        a = self.bn(self.fc(h)) * prior
        return F.softmax(a, dim=-1)


class _TabNetEncoder(nn.Module):
    def __init__(self, input_dim: int, step_dim: int,
                 n_steps: int, gamma: float = 1.3):
        super().__init__()
        self.n_steps  = n_steps
        self.gamma    = gamma
        self.input_bn = nn.BatchNorm1d(input_dim, momentum=0.02)

        self.shared_fc  = nn.Linear(input_dim, step_dim * 2, bias=False)
        self.initial_ft = _TabNetFeatTransformer(input_dim, step_dim, self.shared_fc)

        self.steps = nn.ModuleList([
            nn.ModuleDict({
                "feat": _TabNetFeatTransformer(input_dim, step_dim, self.shared_fc),
                "attn": _TabNetAttentiveTransformer(step_dim, input_dim),
            })
            for _ in range(n_steps)
        ])

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x_bn         = self.input_bn(x)
        h            = self.initial_ft(x_bn)
        prior        = torch.ones_like(x_bn)
        step_outputs = []

        for step in self.steps:
            M     = step["attn"](h, prior)
            prior = prior * (self.gamma - M)
            h     = step["feat"](M * x_bn)
            step_outputs.append(F.relu(h))

        final = torch.stack(step_outputs, dim=0).mean(dim=0)
        return final, step_outputs


class _OfficialTabNetBackend:
    """Wrapper around pytorch_tabnet.pretraining.TabNetPretrainer."""

    def __init__(self, input_dim, step_dim, n_steps, gamma,
                 mask_ratio, epochs, batch_size, lr, seed):
        self._cfg = dict(n_steps=n_steps, step_dim=step_dim)
        self._pt  = _TabNetPretrainer(
            n_d=step_dim, n_a=step_dim,
            n_steps=n_steps, gamma=gamma,
            cat_idxs=[], cat_dims=[],
            optimizer_fn=torch.optim.Adam,
            optimizer_params={"lr": lr},
            mask_type="softmax",
            verbose=0, seed=seed,
        )
        self._meta = dict(mask_ratio=mask_ratio, epochs=epochs,
                          batch_size=batch_size)

    def fit(self, X_train: np.ndarray):
        m = self._meta
        self._pt.fit(
            X_train,
            max_epochs=m["epochs"],
            patience=m["epochs"],
            batch_size=m["batch_size"],
            virtual_batch_size=m["batch_size"],
            pretraining_ratio=m["mask_ratio"],
        )

    def _extract_all(
        self, X: np.ndarray, batch_size: int = 512
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        n_steps  = self._cfg["n_steps"]
        enc      = self._pt.network.encoder
        step_bufs: List[List[np.ndarray]] = [[] for _ in range(n_steps)]
        handles  = []

        for k in range(n_steps):
            def make_hook(idx):
                def hook(mod, inp, out):
                    step_bufs[idx].append(F.relu(out).detach().cpu().numpy())
                return hook
            handles.append(
                enc.step_feature_list[k].register_forward_hook(make_hook(k))
            )

        enc.eval()
        final_bufs: List[np.ndarray] = []
        loader = DataLoader(TensorDataset(to_tensor(X)),
                            batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for (xb,) in loader:
                out, _ = enc(xb.to(DEVICE))
                final_bufs.append(out.cpu().numpy())

        for h in handles:
            h.remove()

        final = np.concatenate(final_bufs, axis=0)
        steps = [np.concatenate(b, axis=0) for b in step_bufs]
        return final, steps


class TabNetModel(BaseSSLModel):
    """
    TabNet with self-supervised pretraining (Arik & Pfister 2021).

    Custom faithful implementation:
      - Shared + step-specific GLU Feature Transformer
      - Attentive Transformer with prior-scale update (gamma)
      - Softmax feature selection mask (sparsemax approximation)
    Pretraining: mask p_mask features, reconstruct via decoder.

    If pytorch-tabnet is installed, pretraining uses the official
    TabNetPretrainer; layer extraction uses forward hooks on step_feature_list.

    Layer names: step_1 ... step_N, final_latent
    """

    name = "tabnet"

    ARCH_CONFIGS = {
        "tabnet_small":  {"step_dim": 16, "n_steps": 3},
        "tabnet_medium": {"step_dim": 32, "n_steps": 4},
        "tabnet_large":  {"step_dim": 64, "n_steps": 5},
    }

    def __init__(
        self,
        input_dim: int,
        arch_name: str    = "tabnet_medium",
        mask_ratio: float = 0.2,
        gamma: float      = 1.3,
        epochs: int       = 50,
        batch_size: int   = 256,
        lr: float         = 1e-3,
        seed: int         = 42,
    ):
        arch = self.ARCH_CONFIGS[arch_name]
        config = dict(arch_name=arch_name, mask_ratio=mask_ratio, gamma=gamma,
                      epochs=epochs, batch_size=batch_size, lr=lr, **arch)
        super().__init__(input_dim, config, seed)
        self.arch_name = arch_name

        if _HAS_PYTORCH_TABNET:
            log.info(f"[TabNet-{arch_name}] Using official pytorch-tabnet.")
            self._official = _OfficialTabNetBackend(
                input_dim, arch["step_dim"], arch["n_steps"], gamma,
                mask_ratio, epochs, batch_size, lr, seed,
            )
            self.encoder    = None
            self.recon_head = None
        else:
            log.info(f"[TabNet-{arch_name}] Using faithful custom implementation.")
            self._official  = None
            self.encoder    = _TabNetEncoder(
                input_dim, arch["step_dim"], arch["n_steps"], gamma
            ).to(self.device)
            self.recon_head = nn.Linear(arch["step_dim"], input_dim).to(self.device)

    # ------------------------------------------------------------------
    def fit_ssl(self, X_train: np.ndarray) -> None:
        set_seed(self.seed)
        if self._official is not None:
            log.info(f"[TabNet-{self.arch_name}] Official pretraining "
                     f"{self.config['epochs']} epochs ...")
            self._official.fit(X_train)
            self._fitted = True
            return

        loader = make_loader(X_train, self.config["batch_size"])
        params = list(self.encoder.parameters()) + list(self.recon_head.parameters())
        optim  = torch.optim.Adam(params, lr=self.config["lr"])
        p_mask = self.config["mask_ratio"]

        log.info(f"[TabNet-{self.arch_name}] Custom pretraining "
                 f"{self.config['epochs']} epochs ...")
        for epoch in range(1, self.config["epochs"] + 1):
            self.encoder.train(); self.recon_head.train()
            total = 0.0
            for (x,) in loader:
                x    = x.to(self.device)
                mask = torch.rand_like(x) < p_mask
                x_in = x.masked_fill(mask, 0.0)
                final, _ = self.encoder(x_in)
                x_hat    = self.recon_head(final)
                loss = (F.mse_loss(x_hat[mask], x[mask]) if mask.any()
                        else x_hat.new_zeros(1).mean())
                optim.zero_grad(); loss.backward(); optim.step()
                total += loss.item()
            if epoch % max(1, self.config["epochs"] // 5) == 0 or epoch == 1:
                log.info(f"  epoch {epoch:3d}/{self.config['epochs']}  loss={total/len(loader):.4f}")
        self._fitted = True

    def list_available_layers(self) -> List[str]:
        n = self.config["n_steps"]
        return [f"step_{i+1}" for i in range(n)] + ["final_latent"]

    @torch.no_grad()
    def extract_embeddings(self, X: np.ndarray, layer_name: str) -> np.ndarray:
        if self._official is not None:
            final, steps = self._official._extract_all(X)
            if layer_name == "final_latent":
                return final
            idx = int(layer_name.split("_")[1]) - 1
            return steps[idx]

        self.encoder.eval()
        out_list = []
        for (x,) in make_loader(X, batch_size=512, shuffle=False):
            x              = x.to(self.device)
            final, step_outs = self.encoder(x)
            if layer_name == "final_latent":
                out = final
            else:
                idx = int(layer_name.split("_")[1]) - 1
                out = step_outs[idx]
            out_list.append(out.cpu().numpy())
        return np.concatenate(out_list, axis=0)

    def _get_state(self):
        if self._official is not None:
            return {"official_pt": self._official._pt, "config": self.config}
        return {"encoder":    self.encoder.state_dict(),
                "recon_head": self.recon_head.state_dict(),
                "config":     self.config}

    def _set_state(self, state):
        if self._official is not None:
            self._official._pt = state["official_pt"]
            return
        self.encoder.load_state_dict(state["encoder"])
        self.recon_head.load_state_dict(state["recon_head"])


# ===========================================================================
# Registry and factory
# ===========================================================================

SSL_MODEL_REGISTRY: Dict[str, Dict[str, type]] = {
    "vime":   {n: VIMEModel   for n in VIMEModel.ARCH_CONFIGS},
    "scarf":  {n: SCARFModel  for n in SCARFModel.ARCH_CONFIGS},
    "saint":  {n: SAINTModel  for n in SAINTModel.ARCH_CONFIGS},
    "subtab": {n: SubTabModel for n in SubTabModel.ARCH_CONFIGS},
    "dae":    {n: DAEModel    for n in DAEModel.ARCH_CONFIGS},
    "tabnet": {n: TabNetModel for n in TabNetModel.ARCH_CONFIGS},
}


def build_ssl_model(
    ssl_family: str,
    arch_name: str,
    input_dim: int,
    epochs: int     = 50,
    batch_size: int = 256,
    lr: float       = 1e-3,
    seed: int       = 42,
) -> BaseSSLModel:
    """Factory: instantiate an SSL model by family + architecture name."""
    family_cls = {
        "vime":   VIMEModel,
        "scarf":  SCARFModel,
        "saint":  SAINTModel,
        "subtab": SubTabModel,
        "dae":    DAEModel,
        "tabnet": TabNetModel,
    }
    cls = family_cls[ssl_family]
    if ssl_family == "saint":
        batch_size = min(batch_size, 64)
    return cls(
        input_dim=input_dim,
        arch_name=arch_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
    )


def get_all_architectures(ssl_family: str) -> List[str]:
    """Return all architecture names registered for a given SSL family."""
    return {
        "vime":   list(VIMEModel.ARCH_CONFIGS.keys()),
        "scarf":  list(SCARFModel.ARCH_CONFIGS.keys()),
        "saint":  list(SAINTModel.ARCH_CONFIGS.keys()),
        "subtab": list(SubTabModel.ARCH_CONFIGS.keys()),
        "dae":    list(DAEModel.ARCH_CONFIGS.keys()),
        "tabnet": list(TabNetModel.ARCH_CONFIGS.keys()),
    }[ssl_family]
