# =========================================================
# CIFAR-10 + ResNet18: layer-wise Hessian trace (HAWQ-V2風)
# =========================================================
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from model import make_model
from train_various_data_net import quick_train
# ------------------------------------------------------------
# 3) ヘルパー: 各「レイヤ」（Conv/Linear）ごとのパラメタ群を抽出
#    ここでは Conv2d/Linear を1単位レイヤとして扱う（HAWQ-V2の想定に近い）
# ------------------------------------------------------------
def group_params_by_layer(model: nn.Module):
    # module名 -> module
    name2mod = dict(model.named_modules())
    # module名 -> そのmoduleに属する trainable parameters [(pname, param), ...]
    buckets = {}

    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # "layer1.0.conv1.weight" -> "layer1.0.conv1"
        mname = pname.rsplit('.', 1)[0]
        mod = name2mod.get(mname, None)
        # Conv/Linear のみ対象（BatchNorm等は除外）
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            buckets.setdefault(mname, []).append((pname, p))
    return buckets  # {layer_name: [(pname, p), ...], ...}

def hvp_layer(model, loss_fn, data_loader, layer_params, v_flat,
              device='cuda', num_batches=2):
    """
    Returns H_l v for 'layer_params' only, averaged over 'num_batches' batches.
    layer_params: List[Tensor] そのレイヤの（weight/biasなど）テンソル
    v_flat: 1D vector for this layer (sum numel == layer params size)
    """
    # v をパラメタ形状に分割
    shapes = [p.shape for p in layer_params]
    sizes  = [p.numel() for p in layer_params]
    v_parts = list(v_flat.split(sizes))
    v_tensors = [vp.view(sh).to(device) for vp, sh in zip(v_parts, shapes)]

    out_flat = torch.zeros_like(v_flat, device=device)
    count = 0

    # 二階微分なので no_grad は使わない
    for xb, yb in data_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

        # 前方 + loss
        model.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)

        # 一階勾配（このレイヤのパラメタだけに対して）
        grads = torch.autograd.grad(loss, layer_params, create_graph=True, retain_graph=True)

        # v に沿った方向微分（ベクトルヤコビアン積）→ H_l v
        Hv = torch.autograd.grad(
            grads, layer_params, grad_outputs=v_tensors, retain_graph=False
        )

        Hv_flat = torch.cat([h.reshape(-1) for h in Hv]).detach()
        out_flat += Hv_flat
        count += 1
        if count >= num_batches:
            break

    return out_flat / max(count, 1)

# ------------------------------------------------------------
# 5) Hutchinson の確率的トレース推定（Rademacher）
#    tr(H_l) ≈ E_z [ z^T H_l z ],  z_i ∈ {±1}
# ------------------------------------------------------------
def hessian_trace_for_layer(model, loss_fn, data_loader, layer_params,
                            num_trace_samples=64, num_batches=2, device='cuda'):
    dim = sum(p.numel() for p in layer_params)
    trace_est = 0.0
    for _ in range(num_trace_samples):
        # Rademacher ±1 ベクトル
        z = torch.empty(dim, device=device).bernoulli_(0.5).mul_(2).sub_(1).float()
        Hz = hvp_layer(model, loss_fn, data_loader, layer_params, z,
                       device=device, num_batches=num_batches)
        trace_est += torch.dot(z, Hz).item()
    return trace_est / num_trace_samples

def train(device):
    torch.manual_seed(0)

    # ----------------------------
    # 1) CIFAR-10 DataLoaders
    # ----------------------------
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2470, 0.2435, 0.2616)),
    ])
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                            (0.2470, 0.2435, 0.2616)),
    ])

    train_set = datasets.CIFAR10(root='./data', train=True, download=True,
                                transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True,
                            num_workers=4, pin_memory=True)

    model = make_model().to(device)
    model=quick_train(model, train_loader, epochs=1, lr=0.1)
    model.eval()  # BN/Dropout固定（2階微分安定化のため）
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    return model,loss_fn,train_loader

def hessian_trase_layerwise(device,model,loss_fn,train_loader,show=True):
    layer_param_groups = group_params_by_layer(model)

    # ------------------------------------------------------------
    # 4) レイヤ単位の HVP（Hessian-vector product）
    #    与えた "そのレイヤのパラメタ群" に対する Hv を返す
    # ------------------------------------------------------------
    @torch.no_grad()
    def _flatten_params(params):
        return torch.cat([p.detach().reshape(-1) for p in params])

    # ------------------------------------------------------------
    # 6) 全レイヤについて計算し、重要度 (= trace / #params) も併記
    # ------------------------------------------------------------
    results = []  # list of dict
    num_trace_samples = 64   # 精度と計算時間のトレードオフ（例: 32〜128）
    num_batches_for_hvp = 2  # データ平均の近似（増やすと安定だが重い）

    for lname, pairs in layer_param_groups.items():
        params = [p for (_, p) in pairs]
        n_params = sum(p.numel() for p in params)

        trace = hessian_trace_for_layer(
            model, loss_fn, train_loader, params,
            num_trace_samples=num_trace_samples,
            num_batches=num_batches_for_hvp,
            device=device
        )
        results.append({
            "layer": lname, "num_params": n_params,
            "trace": trace, "trace_per_param": trace / n_params
        })

    # 重要度順にソート（trace_per_param 降順）
    results = sorted(results, key=lambda x: x["trace_per_param"], reverse=True)

    # 表示
    if(show):
        print(f"[HAWQ-V2 style] Layer-wise Hessian trace on CIFAR-10/ResNet18")
        print(f"(trace samples={num_trace_samples}, batches per HVP={num_batches_for_hvp})")
        for r in results:
            print(f"{r['layer']:28s}  #params={r['num_params']:7d}  "
                f"trace={r['trace']:.3e}  trace/param={r['trace_per_param']:.3e}")
    
    # ------------------------------------------------------------
    # 7) （オプション）簡易ビット割当のヒント
    #    大きい trace/param の層ほど高ビットにする（HAWQ系の典型方針）
    # ------------------------------------------------------------
    # 例: 上位30%→8bit、中位40%→6bit、下位30%→4bit
    cuts = [0.3, 0.7]  # 0~0.3, 0.3~0.7, 0.7~1.0
    L = len(results)
    bound1 = int(L * cuts[0])
    bound2 = int(L * cuts[1])
    for i, r in enumerate(results):
        if i < bound1:   r["suggested_bits"] = 8
        elif i < bound2: r["suggested_bits"] = 6
        else:            r["suggested_bits"] = 4

    print("\n[Suggested mixed-precision (toy rule)]")
    for r in results:
        print(f"{r['layer']:28s}  bits={r['suggested_bits']}  "
            f"trace/param={r['trace_per_param']:.3e}")
        
    return results

def traineval_layerwise(show=True):
    import json
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device,",device)
    model,loss_fn,train_loader=train()    
    results=hessian_trase_layerwise(device,model,loss_fn,train_loader,show)
    with oepn("layerwire_result.json","w") as fp:
        json.dump(results,fp)

if __name__=="__main__":
    traineval_layerwise()