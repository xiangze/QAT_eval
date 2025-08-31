"""
Evaluate multiple layerwise quantization allocation methods on a trained network.
Methods:
  - HAWQ2_fisher (greedy via empirical Fisher)
  - OT_HAWQ_like (Sinkhorn OT with size-weighted marginals)
  - OT_Fisher_Critical (same API; if no critical classes -> same as OT_HAWQ_like)
  - DiffSinkhornDynamic (differentiable Sinkhorn with STE mixture + short QAT)
  - SinkhornMCKPDynamic (Chen et al. quadratic cost + short QAT)

Outputs:
  - Pre-quantization (FP32) accuracy
  - For each method: accuracy, quantized size (MB), mean bits, per-layer assignment CSV
"""

import argparse, json, os, math, csv, time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# =========================
# Utils: layers & quant
# =========================

def iter_quant_layers(model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            yield name, m

def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def quantize_weight_per_tensor_symmetric(w: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 32:
        return w.clone()
    qmax = (1 << (bits - 1)) - 1
    scale = w.detach().abs().max() / (qmax + 1e-12)
    scale = max(scale.item(), 1e-12)
    q = torch.clamp(torch.round(w / scale), min=-qmax-1, max=qmax)
    return q * scale

def apply_assignment_inplace(model: nn.Module, assignment: Dict[str, int]) -> Dict[str, torch.Tensor]:
    """
    assignment: {layer_name: bit}
    Returns backup for restoration.
    """
    backup = {}
    with torch.no_grad():
        for name, m in iter_quant_layers(model):
            b = assignment.get(name, None)
            if b is None: continue
            if hasattr(m, "weight") and m.weight is not None:
                backup[f"{name}.weight"] = m.weight.detach().clone()
                q = quantize_weight_per_tensor_symmetric(m.weight.data, b)
                m.weight.data.copy_(q)
    return backup

def restore_from_backup(model: nn.Module, backup: Dict[str, torch.Tensor]):
    with torch.no_grad():
        for key, tensor in backup.items():
            mod = model
            path, p = key.rsplit(".", 1)
            for token in path.split("."):
                if token.isdigit(): mod = mod[int(token)]
                else: mod = getattr(mod, token)
            getattr(mod, p).data.copy_(tensor)

def model_fp32_size_mb(model: nn.Module) -> float:
    n_params = sum(p.numel() for p in model.parameters())
    return n_params * 32 / 8 / 1e6

def quantized_weight_size_mb(model: nn.Module, assignment: Dict[str, int], include_bias_fp32=True) -> float:
    """
    Size = sum_i (weight_params_i * b_i) + (optional) bias in 32-bit.
    """
    total_bits = 0
    for name, m in iter_quant_layers(model):
        b = assignment.get(name, None)
        if b is None: b = 32
        w_params = m.weight.numel() if hasattr(m, "weight") and m.weight is not None else 0
        total_bits += w_params * b
        if include_bias_fp32 and hasattr(m, "bias") and m.bias is not None:
            total_bits += m.bias.numel() * 32
    return total_bits / 8 / 1e6

def save_assignment_csv(path: str, assignment: Dict[str, int], model: nn.Module):
    rows = []
    for name, m in iter_quant_layers(model):
        rows.append([name, param_count(m), assignment.get(name, 32)])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer_name", "param_count", "bits"])
        w.writerows(rows)

def mean_bits(assignment: Dict[str, int], model: nn.Module) -> float:
    num = 0
    den = 0
    for name, m in iter_quant_layers(model):
        b = assignment.get(name, 32)
        n = m.weight.numel() if hasattr(m, "weight") and m.weight is not None else 0
        num += b * n
        den += n
    return num / max(1, den)

# =========================
# Data & model builders
# =========================

def build_cifar10_loaders(batch=128, num_workers=4, train_aug=False):
    from torchvision import datasets, transforms
    tfm_train = [
        transforms.Resize(224),
        transforms.RandomHorizontalFlip() if train_aug else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]
    tfm_train = [t for t in tfm_train if not isinstance(t, transforms.Lambda)]
    tfm_train = transforms.Compose(tfm_train)

    tfm_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    train = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm_train)
    test  = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm_test)
    return DataLoader(train, batch_size=batch, shuffle=True, num_workers=num_workers, pin_memory=True), \
           DataLoader(test,  batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True), 10

def build_imagenet_val_loader(val_dir: str, batch=128, num_workers=8):
    from torchvision import datasets, transforms
    tfm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    ds = datasets.ImageFolder(val_dir, transform=tfm)
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True), 1000

def build_model(name: str, num_classes: int, pretrained_tv: bool, device: torch.device):
    from torchvision import models
    name = name.lower()
    if name == "resnet18":
        if pretrained_tv:
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            if m.fc.out_features != num_classes:
                m.fc = nn.Linear(m.fc.in_features, num_classes)
        else:
            m = models.resnet18(num_classes=num_classes)
    elif name == "mobilenet_v2":
        if pretrained_tv:
            m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            if m.classifier[-1].out_features != num_classes:
                m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        else:
            m = models.mobilenet_v2(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {name}")
    return m.to(device)

def train_cifar_quick(model: nn.Module, train_loader, test_loader, device, epochs=2, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
        acc = evaluate_top1(model, test_loader, device)
        print(f"[train] epoch {ep} CIFAR10 acc={acc:.2f}%")
    return model

@torch.no_grad()
def evaluate_top1(model: nn.Module, loader: DataLoader, device) -> float:
    model.eval()
    corr = 0
    total = 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        corr += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * corr / max(1,total)

# =========================
# Sensitivity (Empirical Fisher)
# =========================

def empirical_fisher_sensitivity(model: nn.Module, loader: DataLoader, device, batches=1) -> Dict[str, float]:
    ce = nn.CrossEntropyLoss()
    model.train()
    sens = {name: 0.0 for name, _ in iter_quant_layers(model)}
    counts = {k: 0 for k in sens.keys()}
    it = iter(loader)
    for _ in range(batches):
        try: x,y = next(it)
        except StopIteration: break
        x,y = x.to(device), y.to(device)
        for p in model.parameters():
            if p.grad is not None: p.grad.zero_()
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        for name, m in iter_quant_layers(model):
            g2, n = 0.0, 0
            for p in m.parameters():
                if p.grad is None: continue
                g2 += torch.sum(p.grad.detach()**2).item()
                n += p.numel()
            if n>0:
                sens[name] += g2 / n
                counts[name] += 1
    eps = 1e-12
    for k in sens.keys():
        if counts[k] > 0: sens[k] /= counts[k]
        sens[k] = max(sens[k], eps)
    ssum = sum(sens.values())
    for k in sens.keys(): sens[k] /= ssum
    return sens  # sum=1

# =========================
# Method A: HAWQ2_fisher (greedy)
# =========================

def hawq2_fisher_allocate(model: nn.Module, loader: DataLoader, device,
                          bits: List[int], avg_bits: float, batches=1) -> Dict[str,int]:
    sens = empirical_fisher_sensitivity(model, loader, device, batches=batches)
    items = []
    for name, m in iter_quant_layers(model):
        n = m.weight.numel() if hasattr(m,"weight") and m.weight is not None else 0
        items.append((name, sens[name], n))
    # importance score: s_i * n_i
    items.sort(key=lambda x: x[1]*x[2], reverse=True)
    # target param-mass per bit
    total = sum(n for _,_,n in items)
    # derive column fractions via max-entropy under E[b]=avg_bits
    b = torch.tensor(bits, dtype=torch.float64)
    lo, hi = -50.0, 50.0
    def mean_bits(lmb):
        w = torch.exp(lmb*b); q = w/w.sum(); return float((q*b).sum())
    for _ in range(80):
        mid = 0.5*(lo+hi); m = mean_bits(mid)
        if m < avg_bits: lo = mid
        else: hi = mid
    lam = 0.5*(lo+hi); w = torch.exp(lam*b); q = (w/w.sum()).tolist()
    capacities = [int(round(total * qi)) for qi in q]  # in params
    # greedy fill: assign larger bits to more important layers until capacity
    assignment = {}
    used = [0]*len(bits)
    for name, s, n in items:
        # choose best bit that does not exceed capacity too much
        best_j, best_score = 0, -1e18
        for j, bj in enumerate(bits):
            overflow = max(0, used[j] + n - capacities[j])
            score = bj*1e-3 - 1e-6*overflow  # favor large bits but penalize overflow
            if score > best_score: best_score, best_j = score, j
        assignment[name] = bits[best_j]
        used[best_j] += n
    return assignment

# =========================
# Method B: OT_HAWQ_like
# =========================

def sinkhorn_log(cost: torch.Tensor, a: torch.Tensor, b: torch.Tensor, eps=0.02, iters=400):
    K = -cost/eps
    f = torch.zeros(cost.size(0), device=cost.device)
    g = torch.zeros(cost.size(1), device=cost.device)
    def lse(x, dim=-1):
        m,_ = torch.max(x, dim=dim, keepdim=True)
        return (m + torch.log(torch.sum(torch.exp(x-m), dim=dim, keepdim=True))).squeeze(dim)
    log_a = torch.log(a+1e-40); log_b = torch.log(b+1e-40)
    for _ in range(iters):
        f = log_a - lse(K + g.unsqueeze(0), dim=1)
        g = log_b - lse((K + f.unsqueeze(1)).transpose(0,1), dim=1)
    P = torch.exp(K + f.unsqueeze(1) + g.unsqueeze(0))
    return P / (P.sum() + 1e-40)

def build_cost_sens_size_pow2(model: nn.Module, sens: Dict[str,float], bits: List[int], device) -> Tuple[torch.Tensor,List[str],List[int]]:
    names, sizes = [], []
    for name, m in iter_quant_layers(model):
        names.append(name)
        sizes.append(m.weight.numel())
    s = torch.tensor([sens[nm] for nm in names], dtype=torch.float64, device=device)
    n = torch.tensor(sizes, dtype=torch.float64, device=device)
    err = torch.tensor([2.0**(-2*b) for b in bits], dtype=torch.float64, device=device)
    C = (s.unsqueeze(1)*n.unsqueeze(1))*err.unsqueeze(0)
    return C.to(torch.float32), names, sizes

def maxent_b_dist(bits: List[int], target_avg: float) -> List[float]:
    b = torch.tensor(bits, dtype=torch.float64)
    lo,hi = -50.0,50.0
    for _ in range(80):
        mid = 0.5*(lo+hi)
        w = torch.exp(mid*b); q=w/w.sum(); m=float((q*b).sum())
        if m < target_avg: lo=mid
        else: hi=mid
    lam = 0.5*(lo+hi); w=torch.exp(lam*b); q=(w/w.sum()).tolist()
    return q

def size_aware_rounding(P: torch.Tensor, sizes: List[int], bits: List[int], col_fracs: List[float]) -> Dict[str,int]:
    L,B = P.shape
    total = sum(sizes)
    capacities = [int(round(total*f)) for f in col_fracs]
    items = []
    for i in range(L):
        for j in range(B):
            items.append((float(P[i,j].item()), i, j))
    items.sort(reverse=True, key=lambda x:x[0])
    assigned = [-1]*L
    used = [0]*B
    for score,i,j in items:
        if assigned[i]!=-1: continue
        if used[j] + sizes[i] <= capacities[j]:
            assigned[i]=j; used[j]+=sizes[i]
        if all(x!=-1 for x in assigned): break
    # fallback
    for i in range(L):
        if assigned[i]==-1:
            j = max(range(B), key=lambda jb: P[i,jb].item())
            assigned[i]=j
    return assigned

def ot_hawq_like_allocate(model: nn.Module, loader: DataLoader, device, bits: List[int],
                          avg_bits: float, sens_batches=1, eps=0.02, iters=400) -> Dict[str,int]:
    sens = empirical_fisher_sensitivity(model, loader, device, batches=sens_batches)
    C, names, sizes = build_cost_sens_size_pow2(model, sens, bits, device)
    n = torch.tensor(sizes, dtype=torch.float32, device=device); a = n/n.sum()
    col_fracs = maxent_b_dist(bits, avg_bits)
    b = torch.tensor(col_fracs, dtype=torch.float32, device=device)
    P = sinkhorn_log(C, a, b, eps, iters)  # [L,B]
    idxs = size_aware_rounding(P, sizes, bits, col_fracs)
    return {nm: bits[idxs[i]] for i,nm in enumerate(names)}

# =========================
# Method C: OT_Fisher_Critical (同 API)
#   クリティカル変換省略時はOT_HAWQ_likeと同じ
# =========================

def ot_fisher_critical_allocate(model: nn.Module, loader: DataLoader, device, bits: List[int],
                                avg_bits: float, sens_batches=1,
                                critical_ids: Optional[List[int]] = None) -> Dict[str,int]:
    # For simplicity: if no critical_ids given -> same as OT_HAWQ_like
    return ot_hawq_like_allocate(model, loader, device, bits, avg_bits, sens_batches=sens_batches)

# =========================
# Method D/E: Differentiable Sinkhorn (dynamic)
#   - Minimal inlined versions from previous prototypes
# =========================

class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return torch.round(x)
    @staticmethod
    def backward(ctx, g): return g

def ste_round(x: torch.Tensor) -> torch.Tensor: return _RoundSTE.apply(x)

def quantize_per_tensor_symmetric_ste(w: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 32: return w
    qmax = (1 << (bits - 1)) - 1
    scale = w.abs().amax() / (qmax + 1e-12)
    inv = 1.0 / (scale + 1e-12)
    z = ste_round(w * inv)
    z = torch.clamp(z, min=-qmax-1, max=qmax)
    return z * scale

def sinkhorn_log_diff(cost, a, b, epsilon=0.02, iters=200):
    K = -cost/epsilon
    f = torch.zeros(cost.size(0), device=cost.device, dtype=cost.dtype)
    g = torch.zeros(cost.size(1), device=cost.device, dtype=cost.dtype)
    def lse(x, dim=-1):
        m,_ = torch.max(x, dim=dim, keepdim=True)
        return (m + torch.log(torch.sum(torch.exp(x-m), dim=dim, keepdim=True))).squeeze(dim)
    log_a = torch.log(a+1e-40).to(cost.dtype); log_b = torch.log(b+1e-40).to(cost.dtype)
    for _ in range(iters):
        f = log_a - lse(K + g.unsqueeze(0), dim=1)
        g = log_b - lse((K + f.unsqueeze(1)).transpose(0,1), dim=1)
    P = torch.exp(K + f.unsqueeze(1) + g.unsqueeze(0))
    return P / (P.sum() + 1e-40)

class DiffAllocator(nn.Module):
    def __init__(self, layer_names: List[str], layer_sizes: List[int], bits: List[int], device):
        super().__init__()
        self.layer_names = layer_names; self.layer_sizes = layer_sizes; self.bits = bits
        self.L, self.B = len(layer_names), len(bits)
        n = torch.tensor(layer_sizes, dtype=torch.float32, device=device)
        self.register_buffer("a", (n/n.sum()).detach())
        self.theta = nn.Parameter(torch.zeros(self.L, self.B, device=device))
        self.phi = nn.Parameter(torch.zeros(self.B, device=device))
        self.register_buffer("sens", torch.full((self.L,), 1.0/self.L, device=device))
        self.register_buffer("err", torch.tensor([2.0**(-2*b) for b in bits], dtype=torch.float32, device=device))

    @torch.no_grad()
    def update_sensitivity(self, s_map: Dict[str,float]):
        s = torch.tensor([s_map[nm] for nm in self.layer_names], dtype=torch.float32, device=self.a.device)
        s = s / (s.sum() + 1e-12); self.sens.copy_(s)

    def build_cost(self):
        n = torch.tensor(self.layer_sizes, dtype=torch.float32, device=self.a.device)
        return (n*self.sens).unsqueeze(1)*self.err.unsqueeze(0)

    def b(self): return F.softmax(self.phi, dim=0)

    def compute_P(self, eps=0.02, iters=200):
        C = self.build_cost()
        P = sinkhorn_log_diff(C - self.theta, self.a, self.b(), eps, iters)
        return P

    def budget_reg(self, P, target_avg, weight):
        bits_t = torch.tensor(self.bits, dtype=P.dtype, device=P.device)
        Eb = torch.sum(self.a.unsqueeze(1) * P * bits_t.unsqueeze(0))
        return weight * (Eb - target_avg)**2

class OTQLinear(nn.Linear):
    def __init__(self, in_f, out_f, bias=True, idx=0, alloc: DiffAllocator=None, bits=None):
        super().__init__(in_f, out_f, bias=bias); self.idx=idx; self.alloc=alloc; self.bits=bits
    def forward(self, x):
        P = self.alloc._P
        probs = P[self.idx]
        mix = 0.0
        for j,b in enumerate(self.bits):
            Wq = quantize_per_tensor_symmetric_ste(self.weight, b)
            mix = mix + probs[j]*Wq
        return F.linear(x, mix, self.bias)

class OTQConv2d(nn.Conv2d):
    def __init__(self, *args, idx=0, alloc: DiffAllocator=None, bits=None, **kw):
        super().__init__(*args, **kw); self.idx=idx; self.alloc=alloc; self.bits=bits
    def forward(self, x):
        P = self.alloc._P
        probs = P[self.idx]
        mix = 0.0
        for j,b in enumerate(self.bits):
            Wq = quantize_per_tensor_symmetric_ste(self.weight, b)
            mix = mix + probs[j]*Wq
        return F.conv2d(x, mix, self.bias, self.stride, self.padding, self.dilation, self.groups)

def wrap_with_mixture_modules(model: nn.Module, alloc: DiffAllocator, bits: List[int]) -> nn.Module:
    idx=0
    def _rep(mod: nn.Module):
        nonlocal idx
        for name, ch in list(mod.named_children()):
            if isinstance(ch, nn.Conv2d):
                q = OTQConv2d(ch.in_channels, ch.out_channels, kernel_size=ch.kernel_size,
                              stride=ch.stride, padding=ch.padding, dilation=ch.dilation,
                              groups=ch.groups, bias=(ch.bias is not None),
                              idx=idx, alloc=alloc, bits=bits)
                q.weight = ch.weight; q.bias=ch.bias
                setattr(mod, name, q); idx+=1
            elif isinstance(ch, nn.Linear):
                q = OTQLinear(ch.in_features, ch.out_features, bias=(ch.bias is not None),
                              idx=idx, alloc=alloc, bits=bits)
                q.weight = ch.weight; q.bias=ch.bias
                setattr(mod, name, q); idx+=1
            else:
                _rep(ch)
    _rep(model); return model

# Chen cost allocator (dynamic)
class ChenAllocator(nn.Module):
    def __init__(self, layer_names: List[str], layer_sizes: List[int], bits: List[int], device):
        super().__init__()
        self.layer_names = layer_names; self.layer_sizes = layer_sizes; self.bits=bits
        self.L, self.B = len(layer_names), len(bits)
        n = torch.tensor(layer_sizes, dtype=torch.float32, device=device)
        self.register_buffer("a", (n/n.sum()).detach())
        self.register_buffer("trH", torch.full((self.L,), 1.0, device=device))
        self.register_buffer("wmax", torch.full((self.L,), 1.0, device=device))
        self.theta = nn.Parameter(torch.zeros(self.L, self.B, device=device))
        self.phi   = nn.Parameter(torch.zeros(self.B, device=device))

    @torch.no_grad()
    def update_trH(self, tr_map: Dict[str,float]):
        vals = [max(float(tr_map[nm]),1e-12) for nm in self.layer_names]
        self.trH.copy_(torch.tensor(vals, device=self.a.device, dtype=torch.float32))

    @torch.no_grad()
    def update_wmax_from_model(self, model: nn.Module):
        vals=[]
        for nm in self.layer_names:
            # traverse
            mod = model
            for tk in nm.split("."):
                if tk.isdigit(): mod=mod[int(tk)]
                else: mod=getattr(mod, tk)
            vals.append(float(mod.weight.detach().abs().max().item())+1e-12)
        self.wmax.copy_(torch.tensor(vals, device=self.a.device, dtype=torch.float32))

    def b(self): return F.softmax(self.phi, dim=0)

    def build_cost(self):
        L,B=self.L,self.B
        tr = self.trH.unsqueeze(1)                   # [L,1]
        w  = self.wmax.unsqueeze(1)                  # [L,1]
        denom = torch.tensor([(1<<b)-1 for b in self.bits], device=self.a.device, dtype=torch.float32).unsqueeze(0)  # [1,B]
        delta = 2.0*w/denom
        sigma2 = (delta*delta)/12.0
        C = 0.5*tr*sigma2
        return C

    def compute_P(self, eps=0.02, iters=200):
        C = self.build_cost()
        return sinkhorn_log_diff(C - self.theta, self.a, self.b(), eps, iters)

    def budget_reg(self, P, target_avg, weight):
        bits_t = torch.tensor(self.bits, dtype=P.dtype, device=P.device)
        Eb = torch.sum(self.a.unsqueeze(1)*P*bits_t.unsqueeze(0))
        return weight * (Eb - target_avg)**2

# Short QAT to learn allocator then harden
def dynamic_sinkhorn_allocate(model_ctor: Callable[[], nn.Module],
                              train_loader: DataLoader, val_loader: DataLoader,
                              device, bits: List[int], avg_bits: float,
                              steps: int = 50, lr: float = 1e-4,
                              chen: bool = False) -> Dict[str,int]:
    base = model_ctor().to(device)
    # meta
    names, sizes = [], []
    for name, m in iter_quant_layers(base):
        names.append(name); sizes.append(m.weight.numel())
    # allocator
    if chen:
        alloc = ChenAllocator(names, sizes, bits, device).to(device)
        # initial trH
        tr_map = estimate_trH_empirical_fisher(base, train_loader, device, batches=2)
        alloc.update_trH(tr_map)
    else:
        alloc = DiffAllocator(names, sizes, bits, device).to(device)
        s_map = empirical_fisher_sensitivity(base, train_loader, device, batches=2)
        alloc.update_sensitivity(s_map)
    # wrap model with mixture modules
    model = model_ctor().to(device)
    if chen:
        # just reuse mixture layers using diff allocator for STE forward;
        # we can still read P from Chen allocator by injecting alloc._P before forward
        pass
    model = wrap_with_mixture_modules(model, alloc if not chen else alloc, bits).to(device)

    ce = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(list(model.parameters()) + list(alloc.parameters()), lr=lr)
    it = iter(train_loader)
    for step in range(steps):
        try: x,y = next(it)
        except StopIteration:
            it = iter(train_loader); x,y = next(it)
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        # refresh stats
        if chen: alloc.update_wmax_from_model(model)
        # compute P
        P = alloc.compute_P(eps=0.02, iters=150)
        alloc._P = P  # inject
        logits = model(x)
        loss = ce(logits, y)
        loss = loss + (alloc.budget_reg(P, avg_bits, weight=1e-3))
        loss.backward()
        opt.step()
        if step % 20 == 0:
            with torch.no_grad():
                bits_t = torch.tensor(bits, device=device, dtype=P.dtype)
                Eb = (alloc.a.unsqueeze(1)*P*bits_t.unsqueeze(0)).sum().item()
            print(f"[dynamic {'Chen' if chen else 'Diff'}] step {step} Eb≈{Eb:.3f}")

    # Harden: take argmax per row
    with torch.no_grad():
        P = alloc.compute_P(eps=0.02, iters=200)
        idxs = P.argmax(dim=1).tolist()
    assignment = {names[i]: bits[idxs[i]] for i in range(len(names))}
    return assignment

# Empirical Fisher trace (sum of grad^2 per layer)
def estimate_trH_empirical_fisher(model: nn.Module, loader: DataLoader, device, batches=1) -> Dict[str,float]:
    ce = nn.CrossEntropyLoss()
    model.train()
    sens = {name: 0.0 for name,_ in iter_quant_layers(model)}
    counts = {k:0 for k in sens.keys()}
    it = iter(loader)
    for _ in range(batches):
        try: x,y = next(it)
        except StopIteration: break
        x,y = x.to(device), y.to(device)
        for p in model.parameters():
            if p.grad is not None: p.grad.zero_()
        logits = model(x); loss = ce(logits, y)
        loss.backward()
        for name, m in iter_quant_layers(model):
            g2 = 0.0
            for p in m.parameters():
                if p.grad is None: continue
                g2 += torch.sum(p.grad.detach()**2).item()
            sens[name] += g2; counts[name]+=1
    out={}
    for k,v in sens.items():
        if counts[k]>0: v/=counts[k]
        out[k]=max(v,1e-12)
    return out

# =========================
# Experiment runner
# =========================

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Data & model
    if args.train_cifar10:
        train_loader, test_loader, ncls = build_cifar10_loaders(batch=args.batch, num_workers=args.workers, train_aug=True)
        model = build_model(args.model, ncls, pretrained_tv=False, device=device)
        model = train_cifar_quick(model, train_loader, test_loader, device, epochs=args.train_epochs, lr=args.lr)
        eval_loader = test_loader
        model_ctor = lambda: build_model(args.model, ncls, pretrained_tv=False, device=device)
    else:
        # torchvision pretrained; need a val loader (ImageNet val path or CIFAR10 test fallback)
        if args.imagenet_val:
            eval_loader, ncls = build_imagenet_val_loader(args.imagenet_val, batch=args.batch, num_workers=args.workers)
            model = build_model(args.model, 1000, pretrained_tv=True, device=device)
            model_ctor = lambda: build_model(args.model, 1000, pretrained_tv=True, device=device)
        else:
            # fallback to CIFAR10 test (head adapted to 10 classes)
            train_loader, test_loader, ncls = build_cifar10_loaders(batch=args.batch, num_workers=args.workers)
            model = build_model(args.model, ncls, pretrained_tv=True, device=device)
            eval_loader = test_loader
            model_ctor = lambda: build_model(args.model, ncls, pretrained_tv=True, device=device)

    # Baseline FP32
    base_acc = evaluate_top1(model, eval_loader, device)
    base_size = model_fp32_size_mb(model)
    print(f"[FP32] acc={base_acc:.2f}%  size={base_size:.2f} MB")

    methods = []
    if "HAWQ2_fisher" in args.methods: methods.append("HAWQ2_fisher")
    if "OT_HAWQ_like" in args.methods: methods.append("OT_HAWQ_like")
    if "OT_Fisher_Critical" in args.methods: methods.append("OT_Fisher_Critical")
    if "DiffSinkhornDynamic" in args.methods: methods.append("DiffSinkhornDynamic")
    if "SinkhornMCKPDynamic" in args.methods: methods.append("SinkhornMCKPDynamic")

    # Iterate methods
    results = []
    for meth in methods:
        print(f"\n=== [{meth}] ===")
        # fresh copy
        qmodel = model_ctor().to(device)
        # build assignment
        if meth == "HAWQ2_fisher":
            assignment = hawq2_fisher_allocate(qmodel, eval_loader, device, args.bits, args.avg_bits, batches=args.sens_batches)
        elif meth == "OT_HAWQ_like":
            assignment = ot_hawq_like_allocate(qmodel, eval_loader, device, args.bits, args.avg_bits,
                                               sens_batches=args.sens_batches, eps=args.sinkhorn_eps, iters=args.sinkhorn_iters)
        elif meth == "OT_Fisher_Critical":
            assignment = ot_fisher_critical_allocate(qmodel, eval_loader, device, args.bits, args.avg_bits,
                                                     sens_batches=args.sens_batches, critical_ids=None)
        elif meth == "DiffSinkhornDynamic":
            assignment = dynamic_sinkhorn_allocate(model_ctor, train_loader if args.train_cifar10 else eval_loader,
                                                   eval_loader, device, args.bits, args.avg_bits,
                                                   steps=args.dynamic_steps, lr=args.dynamic_lr, chen=False)
        elif meth == "SinkhornMCKPDynamic":
            assignment = dynamic_sinkhorn_allocate(model_ctor, train_loader if args.train_cifar10 else eval_loader,
                                                   eval_loader, device, args.bits, args.avg_bits,
                                                   steps=args.dynamic_steps, lr=args.dynamic_lr, chen=True)
        else:
            raise ValueError(meth)

        # apply quantization in-place
        backup = apply_assignment_inplace(qmodel, assignment)
        acc = evaluate_top1(qmodel, eval_loader, device)
        qsize = quantized_weight_size_mb(qmodel, assignment, include_bias_fp32=True)
        mbits = mean_bits(assignment, qmodel)
        print(f"[{meth}] acc={acc:.2f}%  size={qsize:.2f} MB  mean_bits≈{mbits:.2f}")

        # save assignment
        csv_path = os.path.join(args.out_dir, f"{meth}_assignment.csv")
        save_assignment_csv(csv_path, assignment, qmodel)
        results.append({
            "method": meth,
            "acc": acc,
            "size_MB": qsize,
            "mean_bits": mbits,
            "assignment_csv": csv_path
        })
        # restore (not needed since we use fresh qmodel each time)
        # restore_from_backup(qmodel, backup)

    # dump summary
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({
            "fp32": {"acc": base_acc, "size_MB": base_size},
            "bits": args.bits, "avg_bits": args.avg_bits,
            "results": results
        }, f, indent=2)
    print("\n=== Summary ===")
    print(json.dumps({"fp32":{"acc":base_acc,"size_MB":base_size},"results":results}, indent=2))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="resnet18", help="resnet18|mobilenet_v2")
    p.add_argument("--cpu", action="store_true")
    # data options
    p.add_argument("--train_cifar10", action="store_true", help="train on CIFAR10 quickly instead of using torchvision pretrained")
    p.add_argument("--train_epochs", type=int, default=2)
    p.add_argument("--imagenet_val", type=str, default="", help="path to ImageNet val dir if evaluating torchvision pretrained")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)

    # methods to run
    p.add_argument("--methods", type=str, nargs="+",
                   default=["HAWQ2_fisher","OT_HAWQ_like","OT_Fisher_Critical","DiffSinkhornDynamic","SinkhornMCKPDynamic"])

    # quant options
    p.add_argument("--bits", type=int, nargs="+", default=[2,4,6,8])
    p.add_argument("--avg_bits", type=float, default=6.0)
    p.add_argument("--sens_batches", type=int, default=2)

    # sinkhorn
    p.add_argument("--sinkhorn_eps", type=float, default=0.02)
    p.add_argument("--sinkhorn_iters", type=int, default=400)

    # dynamic
    p.add_argument("--dynamic_steps", type=int, default=50)
    p.add_argument("--dynamic_lr", type=float, default=1e-4)

    p.add_argument("--out_dir", type=str, default="./quant_results")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args)
