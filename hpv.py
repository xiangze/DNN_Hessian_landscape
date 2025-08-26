import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ===== 2) モデル（ResNet-18 を 10クラスに調整）=====
def make_model():
    model = models.resnet18(weights=None)   # 事前学習なし。ImageNet重みを使うなら weights='IMAGENET1K_V1' など
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def hvp(model, loss_fn, data_loader, v, device='cuda', num_batches=1):
    """
    Hv を返す。データは num_batches バッチで近似平均。
    v は「フラット化したパラメタ」と同じ形の 1D テンソル。
    """
    model.zero_grad(set_to_none=True)
    params = [p for p in model.parameters() if p.requires_grad]
    # v をパラメタ形状に分割
    shapes = [p.shape for p in params]
    sizes  = [p.numel() for p in params]
    splits = list(v.split(sizes))
    v_params = [s.view(sh) for s, sh in zip(splits, shapes)]

    Hv = [torch.zeros_like(p, device=device) for p in params]
    count = 0

    for (xb, yb) in data_loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(model(xb), yb)
        # 一階勾配
        grads = torch.autograd.grad(loss, params, create_graph=True)
        # v に沿った方向微分（ベクトル・ヤコビアン積）
        grad_v = torch.autograd.grad(
            grads, params, grad_outputs=v_params, retain_graph=False
        )
        # 逐次平均
        for i, g in enumerate(grad_v):
            Hv[i] += g.detach()
        count += 1
        if count >= num_batches:
            break

    Hv = [h / count for h in Hv]
    return torch.cat([h.reshape(-1) for h in Hv])

# ===== 3) 参考：軽いウォームアップ学習（オプション）=====
# Hessian評価は学習済み点のほうが意味があるので、1〜数epochだけ回す例
def quick_train(model, train_loader, epochs=1, lr=0.1):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(train_loader))

    for ep in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            scheduler.step()

def train():
    # ===== 1) CIFAR-10 用データ前処理・DataLoader =====
    # 学習時の標準的なData Augmentation＋正規化（CIFAR-10の平均・分散）
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                            std=(0.2470, 0.2435, 0.2616)),
    ])

    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                            std=(0.2470, 0.2435, 0.2616)),
    ])

    train_set = datasets.CIFAR10(root='./data', train=True, download=True,
                                transform=transform_train)
    test_set  = datasets.CIFAR10(root='./data', train=False, download=True,
                                transform=transform_eval)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True,
                            num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=256, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = make_model().to(device)
    # 例：1epochだけ回す（時間節約のため任意）
    quick_train(model, train_loader, epochs=1, lr=0.1)

    # ===== 4) Hessian/HVP用の損失関数 =====
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    # パラメタ次元
    dim = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # 上位固有対

    params = [p for p in model.parameters() if p.requires_grad]
    dim = sum(p.numel() for p in params)   # 総パラメタ数
    print("Total dim:", dim)
    v = torch.randn(dim, device=device)  
    v = v / v.norm()                     

    evals, evecs = hvp(model, loss_fn, train_loader, v, device='cuda', num_batches=1)
                    #hvp(model, loss_fn, train_loader, dim, k=5, m=100, device=device)
    # ランク推定（しきい値でカウント）
    tol = 1e-6  # スケールに応じて調整（例: loss/バッチ平均の規模）
    rank_est = int((evals.abs() > tol).sum().item())
    return rank_est


if __name__=="__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    train()

