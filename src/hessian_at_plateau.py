# curvature_at_plateau.py
# PyTorch: plateau検知 → Hessianスペクトル/非零固有値数の推定（Lanczos, SLQ, Hutchinson）

import math
from dataclasses import dataclass
from typing import Callable, List, Tuple, Iterable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

# -------------------------
# Utils: flatten/unflatten
# -------------------------
def _params_list(model: nn.Module) -> List[torch.Tensor]:
    return [p for p in model.parameters() if p.requires_grad]

def _flatten(tensors: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors], dim=0)

def _split_like(vec: torch.Tensor, like: List[torch.Tensor]) -> List[torch.Tensor]:
    outs = []
    offset = 0
    for t in like:
        numel = t.numel()
        outs.append(vec[offset:offset+numel].view_as(t))
        offset += numel
    return outs

def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -------------------------
# Hessian-vector product (Pearlmutter)
# -------------------------
def make_hvp(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    batch: Tuple[torch.Tensor, torch.Tensor],
    weight_decay: float = 0.0,
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], int, torch.device, torch.dtype]:
    """
    Returns Hv: R^n -> R^n closure at current params, for the given batch.
    H is the Hessian of: loss_fn(model(x), y) [+ L2 weight_decay].
    """
    model.eval()
    params = _params_list(model)
    device = params[0].device
    dtype = params[0].dtype
    n = sum(p.numel() for p in params)

    x, y = batch
    x = x.to(device=device, dtype=dtype, non_blocking=True)
    y = y.to(device=device, non_blocking=True)

    def Hv(v_flat: torch.Tensor) -> torch.Tensor:
        # split v into each param's shape
        v_list = _split_like(v_flat, params)

        # fresh forward each call to fix the evaluation point
        model.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out, y)

        if weight_decay > 0.0:
            reg = 0.0
            for p in params:
                reg = reg + (p * p).sum()
            loss = loss + 0.5 * weight_decay * reg

        # first grad (create_graph=True to build graph for second-order)
        grad = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

        # inner product <grad, v>
        gv = 0.0
        for g, v in zip(grad, v_list):
            gv = gv + (g * v).sum()

        # second grad: ∂/∂params <grad, v> = H v
        hv_list = torch.autograd.grad(gv, params, retain_graph=False)
        hv_flat = _flatten([h.detach() for h in hv_list])  # detach: Hv is a number vector
        return hv_flat

    return Hv, n, device, dtype

# -------------------------
# Lanczos (with reorthogonalization)
# -------------------------
@dataclass
class LanczosResult:
    alphas: torch.Tensor
    betas: torch.Tensor
    T: torch.Tensor
    evals: torch.Tensor   # eigenvalues of T (Ritz vals)
    evecs: torch.Tensor   # eigenvectors of T
    v0: torch.Tensor      # starting vector (normalized)

def lanczos(
    Hv: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    steps: int,
    device: torch.device,
    dtype: torch.dtype,
    v0: Optional[torch.Tensor] = None,
    full_reorth: bool = True,
    tol: float = 1e-7,
) -> LanczosResult:
    """
    Classic Lanczos to tridiagonal T. Suitable for symmetric H.
    """
    if v0 is None:
        v = torch.randn(dim, device=device, dtype=dtype)
    else:
        v = v0.to(device=device, dtype=dtype)
    v = v / (v.norm() + 1e-12)

    Vs: List[torch.Tensor] = []
    alphas = []
    betas = []

    w = Hv(v)
    alpha = torch.dot(v, w)
    w = w - alpha * v
    if full_reorth and Vs:
        for q in Vs:
            w = w - torch.dot(w, q) * q
    beta = w.norm()
    Vs.append(v.clone())
    alphas.append(alpha)
    betas.append(beta)

    for k in range(1, steps):
        if beta.item() < tol:
            # early termination
            break
        v_next = w / beta
        Vs.append(v_next.clone())

        w = Hv(v_next)
        alpha = torch.dot(v_next, w)
        w = w - alpha * v_next - beta * v
        if full_reorth:
            # Full reorth against all previous Vs
            for q in Vs[:-1]:
                w = w - torch.dot(w, q) * q

        v = v_next
        alphas.append(alpha)
        beta = w.norm()
        betas.append(beta)

    m = len(alphas)
    T = torch.zeros((m, m), device=device, dtype=dtype)
    for i in range(m):
        T[i, i] = alphas[i]
        if i + 1 < m:
            T[i, i+1] = betas[i+1]
            T[i+1, i] = betas[i+1]

    # symmetric small matrix eigendecomp on CPU (robust)
    T_cpu = T.detach().cpu()
    evals, evecs = torch.linalg.eigh(T_cpu)
    return LanczosResult(
        alphas=torch.tensor(alphas, device=device, dtype=dtype),
        betas=torch.tensor(betas, device=device, dtype=dtype),
        T=T,
        evals=evals.to(device=device, dtype=dtype),
        evecs=evecs.to(device=device, dtype=dtype),
        v0=Vs[0],
    )

# -------------------------
# SLQ: spectral density estimate
# -------------------------
@torch.no_grad()
def slq_spectral_density(
    Hv: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    probes: int = 20,
    lanczos_steps: int = 64,
    grid_min: float = -5.0,
    grid_max: float = 5.0,
    grid_points: int = 201,
    sigma: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate spectral density rho(λ) ≈ (n/m) Σ_i Σ_j γ_{ij} * N(λ; θ_{ij}, σ^2)
    where θ are Ritz eigenvalues of T_i, and γ are squared first-component weights.
    Returns (grid, density) where integral density dλ ≈ number of eigenvalues (=dim).
    """
    grid = torch.linspace(grid_min, grid_max, grid_points, device=device, dtype=dtype)
    density = torch.zeros_like(grid)

    n = dim
    norm_const = 1.0 / (sigma * math.sqrt(2.0 * math.pi))

    for _ in range(probes):
        v0 = torch.randint(0, 2, (n,), device=device, dtype=torch.int8)
        v0 = v0 * 2 - 1  # Rademacher in {-1,+1}
        v0 = v0.to(dtype=dtype)
        v0 = v0 / (v0.norm() + 1e-12)

        L = lanczos(Hv, dim=n, steps=lanczos_steps, device=device, dtype=dtype, v0=v0, full_reorth=True)
        theta = L.evals           # [m]
        Q = L.evecs               # [m, m]
        w1_sq = (Q[0, :] ** 2)    # γ_j = (first component)^2

        # kernel mixture: sum_j γ_j * N(grid | θ_j, σ^2)
        # multiply by n (dimension) to get density whose integral ≈ n
        for j in range(theta.numel()):
            diff = grid - theta[j]
            density += (n * w1_sq[j]) * norm_const * torch.exp(-0.5 * (diff / sigma) ** 2)

    density /= probes
    return grid, density

# -------------------------
# Hutchinson: Tr(H), Tr(H^2)
# -------------------------
@torch.no_grad()
def hutchinson_traces(
    Hv: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    probes: int = 50,
) -> Tuple[float, float]:
    """
    Returns (trace(H), trace(H^2)) estimates via Hutchinson with Rademacher probes.
    """
    trH = 0.0
    trH2 = 0.0
    for _ in range(probes):
        v = torch.randint(0, 2, (dim,), device=device, dtype=torch.int8)
        v = (v * 2 - 1).to(dtype=dtype)
        v = v / (v.norm() + 1e-12)

        Hv_v = Hv(v)
        trH += torch.dot(v, Hv_v).item()          # E[v^T H v] = Tr(H)
        trH2 += Hv_v.norm().pow(2).item()         # E[v^T H^2 v] = Tr(H^2)

    trH /= probes
    trH2 /= probes
    return trH, trH2

# -------------------------
# Saddle-ness metrics at plateau
# -------------------------
@dataclass
class CurvatureSummary:
    lambda_max: float
    lambda_min: float
    trace_H: float
    trace_H2: float
    nonzero_count_eps: int
    negative_count_eps: int

def curvature_analysis_at_batch(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    batch: Tuple[torch.Tensor, torch.Tensor],
    weight_decay: float = 0.0,
    lanczos_k: int = 64,
    slq_probes: int = 20,
    hutchinson_probes: int = 50,
    grid_min: float = -5.0,
    grid_max: float = 5.0,
    grid_points: int = 201,
    kernel_sigma: float = 0.05,
    eps: float = 1e-6,
) -> Tuple[CurvatureSummary, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Runs Lanczos/SLQ/Hutchinson on one batch. Returns summary + (grid, density).
    """
    Hv, n, device, dtype = make_hvp(model, loss_fn, batch, weight_decay=weight_decay)

    # 1) Extreme eigenvalues from one Lanczos run (good starting v)
    L = lanczos(Hv, dim=n, steps=lanczos_k, device=device, dtype=dtype, v0=None, full_reorth=True)
    lambda_max = float(L.evals.max().item())
    lambda_min = float(L.evals.min().item())

    # 2) Spectral density via SLQ
    grid, density = slq_spectral_density(
        Hv, dim=n, device=device, dtype=dtype,
        probes=slq_probes, lanczos_steps=lanczos_k,
        grid_min=grid_min, grid_max=grid_max, grid_points=grid_points, sigma=kernel_sigma
    )
    dλ = float((grid[-1] - grid[0]) / (grid.numel() - 1))
    # Estimated counts by integrating density
    mask_nonzero = (grid < -eps) | (grid > eps)
    mask_negative = (grid < -eps)
    nonzero_count = int(torch.clamp((density[mask_nonzero].sum() * dλ), min=0.0).round().item())
    negative_count = int(torch.clamp((density[mask_negative].sum() * dλ), min=0.0).round().item())

    # 3) Hutchinson traces
    trH, trH2 = hutchinson_traces(Hv, dim=n, device=device, dtype=dtype, probes=hutchinson_probes)

    summary = CurvatureSummary(
        lambda_max=lambda_max,
        lambda_min=lambda_min,
        trace_H=trH,
        trace_H2=trH2,
        nonzero_count_eps=nonzero_count,
        negative_count_eps=negative_count,
    )
    return summary, (grid.detach().cpu(), density.detach().cpu())

# -------------------------
# Plateau detector
# -------------------------
class PlateauDetector:
    def __init__(self, window: int = 20, rel_tol: float = 1e-4):
        self.window = window
        self.rel_tol = rel_tol
        self.history: List[float] = []
        self.triggered = False

    def update(self, loss_value: float) -> bool:
        self.history.append(loss_value)
        if self.triggered or len(self.history) < (self.window + 1):
            return False
        old = self.history[-(self.window + 1)]
        new = self.history[-1]
        denom = max(abs(old), 1e-8)
        rel_improve = (old - new) / denom
        if rel_improve < self.rel_tol:
            self.triggered = True
            return True
        return False

# -------------------------
# Example training loop glue
# -------------------------
def train_with_curvature_monitor(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    plateau_window: int = 20,
    plateau_tol: float = 1e-4,
    curvature_conf: dict = None,
    analyze_once: bool = True,
) -> None:
    """
    Example: call during training. When plateau detected, run curvature analysis on the latest batch.
    """
    model.to(device)
    detector = PlateauDetector(window=plateau_window, rel_tol=plateau_tol)
    did_analyze = False
    curvature_conf = curvature_conf or {}

    for epoch in range(1, 1000000):  # your stopping condition elsewhere
        for batch_idx, (x, y) in enumerate(train_loader):
            model.train()
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            loss_val = float(loss.detach().cpu().item())
            if detector.update(loss_val) and (not did_analyze or not analyze_once):
                # run curvature analysis on *this* batch
                batch_for_curv = (x.detach(), y.detach())
                summary, (grid, density) = curvature_analysis_at_batch(
                    model, loss_fn, batch_for_curv, **curvature_conf
                )

                # ---- report ----
                print("\n=== Curvature analysis at plateau ===")
                print(f"dim(params)            : {num_params(model)}")
                print(f"lambda_max (≈spectral) : {summary.lambda_max:+.6e}")
                print(f"lambda_min             : {summary.lambda_min:+.6e}")
                print(f"Tr(H) (Hutchinson)     : {summary.trace_H:+.6e}")
                print(f"Tr(H^2) (Hutchinson)   : {summary.trace_H2:+.6e}")
                print(f"# nonzero(|λ|>{curvature_conf.get('eps',1e-6)}): {summary.nonzero_count_eps}")
                print(f"# negative(|λ|>{curvature_conf.get('eps',1e-6)}): {summary.negative_count_eps}")

                # A lower bound on rank via Frobenius norm and spectral norm
                lambda_max_abs = max(abs(summary.lambda_max), abs(summary.lambda_min))
                if lambda_max_abs > 0:
                    rank_lower = summary.trace_H2 / (lambda_max_abs ** 2)
                    print(f"rank(H) lower bound (Tr(H^2)/||H||_2^2): ~ {rank_lower:.2f}")
                else:
                    print("rank(H) lower bound: undefined (λ_max=0)")

                # Optional: save spectrum estimate for plotting later
                try:
                    import numpy as np
                    np.savez("hessian_spectrum_estimate.npz",
                             grid=grid.numpy(), density=density.numpy())
                    print("Saved spectral density to hessian_spectrum_estimate.npz")
                except Exception as e:
                    print(f"(Could not save spectrum: {e})")

                did_analyze = True

        # you will likely break by epochs or validation
        # break

# -------------------------
# Minimal usage example (pseudo)
# -------------------------
if __name__ == "__main__":
    # Define your model/dataloader/optimizer as usual:
    # Example toy model (replace with your own)
    class SmallNet(nn.Module):
        def __init__(self, in_dim=784, hidden=256, num_classes=10):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, num_classes),
            )
        def forward(self, x):
            # expecting x as [B, in_dim]
            return self.net(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallNet()

    # Dummy data loader (replace with real dataset)
    x = torch.randn(1024, 784)
    y = torch.randint(0, 10, (1024,))
    ds = torch.utils.data.TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=0)

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    curvature_conf = dict(
        weight_decay=0.0,
        lanczos_k=64,
        slq_probes=16,
        hutchinson_probes=32,
        grid_min=-10.0,
        grid_max=10.0,
        grid_points=401,
        kernel_sigma=0.05,
        eps=1e-6,
    )

    train_with_curvature_monitor(
        model, loader, loss_fn, optim, device,
        plateau_window=20, plateau_tol=1e-4,
        curvature_conf=curvature_conf, analyze_once=True
    )
