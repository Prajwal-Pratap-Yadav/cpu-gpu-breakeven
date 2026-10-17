"""
Instrumented traffic audit of one eager DE/rand/1/bin generation.
Every ATen op dispatched during one generation is intercepted; bytes read
(tensor inputs) and written (tensor outputs) are summed. This is the
traffic an unfused eager implementation moves if no intermediate is reused
from cache -- an UPPER bound on DRAM traffic. The fused minimum (each
logical array touched once) is the 13-traversal count, B = 52 B/element.
"""
import torch, json, sys
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

def sphere(z): return (z * z).sum(dim=1)

def de_generation(x, evaluate=sphere):
    NP, D = x.shape; dev = x.device
    idx = torch.randint(0, NP, (3, NP), device=dev)
    v = x[idx[0]] + 0.5 * (x[idx[1]] - x[idx[2]])
    mask = torch.rand(NP, D, device=dev) < 0.9
    mask.scatter_(1, torch.randint(0, D, (NP, 1), device=dev), True)
    u = torch.where(mask, v, x)
    better = (evaluate(u) < evaluate(x)).unsqueeze(1)
    return torch.where(better, u, x)

class Audit(TorchDispatchMode):
    def __init__(self): super().__init__(); self.ops=[]
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        ins, seen = [], set()
        for t in tree_flatten((args, kwargs))[0]:
            if isinstance(t, torch.Tensor):
                key = (t.data_ptr(), t.numel(), t.dtype)
                if key in seen:      # e.g. z*z reads z once, not twice
                    continue
                seen.add(key); ins.append(t)
        if func in (torch.ops.aten.select.int, torch.ops.aten.unsqueeze.default):
            ins = []                 # views: no data movement
        outs = [t for t in tree_flatten(out)[0] if isinstance(t, torch.Tensor)]
        if func in (torch.ops.aten.select.int, torch.ops.aten.unsqueeze.default):
            outs = []
        rb = sum(t.numel()*t.element_size() for t in ins)
        wb = sum(t.numel()*t.element_size() for t in outs)
        # in-place ops: output aliases an input; count as read+write of that tensor
        self.ops.append((str(func), rb, wb))
        return out

def audit(NP, D):
    x = torch.rand(NP, D)
    a = Audit()
    with a:
        de_generation(x)
    N = NP*D
    tot = sum(r+w for _,r,w in a.ops)
    return a.ops, tot/N

if __name__ == "__main__":
    res = {}
    for NP, D in [(4096, 300), (1024, 100), (512, 30)]:
        ops, bpe = audit(NP, D)
        res[f"{NP}x{D}"] = {"ops": len(ops), "bytes_per_element": bpe}
        if NP == 4096:
            print(f"ATen ops per generation: {len(ops)}")
            for f, r, w in ops:
                print(f"  {f:<38} read {r/(NP*D):6.2f} B/el   write {w/(NP*D):6.2f} B/el")
        print(f"NP={NP:<5} D={D:<4} ops={len(ops):<3} traffic = {bpe:6.2f} bytes/element/generation")
    json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "results/traversal_audit.json", "w"), indent=2)
