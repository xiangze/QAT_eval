from typing import List, Dict, Tuple, Iterable, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import qutil 

def empirical_fisher_sensitivity(model: nn.Module, loader: DataLoader, device, batches=1) -> Dict[str, float]:
    ce = nn.CrossEntropyLoss()
    model.train()
    sens = {name: 0.0 for name, _ in qutil.iter_quant_layers(model)}
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
        for name, m in qutil.iter_quant_layers(model):
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


"""
平均ビット制約 E[b]=target_avg の下で最大エントロピー分布 q(b)∝exp(λ b) を二分探索で解く。
"""
def maxent_bit_distribution(bits: List[int], avg_bits: float,lo=-50,hi=50.) -> List[float]:
    b = torch.tensor(bits, dtype=torch.float64)
    def mean_bits(lmb):
        w = torch.exp(lmb*b); q = w/w.sum(); return float((q*b).sum())
    for _ in range(80):
        mid = 0.5*(lo+hi); m = mean_bits(mid)
        if m < avg_bits: lo = mid
        else: hi = mid
    lam = 0.5*(lo+hi); 
    w = torch.exp(lam*b); 
    return  (w/w.sum()).tolist()

# greedy fill: assign larger bits to more important layers until capacity
def greedy_rounding_assignment(capacities,bits,items):
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

def hawq2_fisher_allocate(model: nn.Module, loader: DataLoader, device,
                          bits: List[int], avg_bits: float, batches=1) -> Dict[str,int]:
    sens = empirical_fisher_sensitivity(model, loader, device, batches=batches)
    items =[(name, sens[name], m.weight.numel()) if (hasattr(m,"weight") and m.weight is not None) else (name, sens[name], 0) for name, m in qutil.iter_quant_layers(model) ]
    # importance score: s_i * n_i
    items.sort(key=lambda x: x[1]*x[2], reverse=True)
    # target param-mass per bit
    total = sum(n for _,_,n in items)
    q=maxent_bit_distribution(bits,avg_bits)
    capacities = [int(round(total*qi)) for qi in q]  # in params
    return greedy_rounding_assignment(capacities,bits,items)