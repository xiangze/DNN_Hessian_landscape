import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import hpv

def lanczos_topk(model, loss_fn, data_loader, dim, k=10, m=80, device='cuda'):
    """
    k 本の上位固有値を Lanczos で近似（m は反復長）。
    返り値: evals[k], evecs[dim, k]
    """
    Q = []
    alpha = []
    beta = []

    q = torch.randn(dim, device=device); q /= q.norm()
    Q.append(q)
    b_prev = 0.0

    for j in range(m):
        z = hpv.hvp(model, loss_fn, data_loader, Q[-1], device=device)
        a = torch.dot(Q[-1], z).item()
        alpha.append(a)
        if j == 0:
            r = z - a * Q[-1]
        else:
            r = z - a * Q[-1] - b_prev * Q[-2]

        # 再直交化（数値安定用に1回だけ実施）
        for q_old in Q:
            r -= torch.dot(r, q_old) * q_old

        b = r.norm().item()
        beta.append(b)
        if b < 1e-10:  # 収束
            break
        q = r / b
        Q.append(q)
        b_prev = b

    T = torch.zeros(len(Q), len(Q), device=device)
    for i in range(len(alpha)):
        T[i, i] = alpha[i]
    for i in range(len(beta)-1):
        T[i, i+1] = beta[i+1]
        T[i+1, i] = beta[i+1]

    evals, U = torch.linalg.eigh(T)  # tri-diagonal の固有分解
    idx = torch.argsort(evals, descending=True)[:k]
    evals = evals[idx]
    U = U[:, idx]

    Qmat = torch.stack(Q, dim=1)        # [dim, m_eff]
    evecs = Qmat @ U                    # Ritz ベクトル [dim, k]
    return evals, evecs

i
def make_model():
    model = models.resnet18(weights=None)   # 事前学習なし。ImageNet重みを使うなら weights='IMAGENET1K_V1' など
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

# ===== 3) 参考：軽いウォームアップ学習（オプション）=====
# Hessian評価は学習済み点のほうが意味があるので、1〜数epochだけ回す例
def quick_train(model, train_loader, device,epochs=1, lr=0.1):
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

def train(device,vecnum=5):
# ===== 0) 基本セットアップ =====
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)

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
    evals, evecs = lanczos_topk(model, loss_fn, train_loader, dim, k=vecnum, m=100, device=device)
    # ランク推定（しきい値でカウント）
    tol = 1e-6  # スケールに応じて調整（例: loss/バッチ平均の規模）
    rank_est = int((evals.abs() > tol).sum().item())

    return evals, evecs,rank_est

if __name__=="__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vecnum=5
    evals, evecs,rank_est=train(device,vecnum)
    print(f"上位{vecnum}個")
    print(f"rank{rank_est}")
    print(f"固有値{evals}")
    print(f"固有ベクトル{evecs}")