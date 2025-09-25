import torch
from typing import Dict, Literal, Tuple

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
    return model

def simple_train(device,epochs=2):
    torch.manual_seed(0)

    # ===== 1) CIFAR-10 用データ前処理・DataLoader =====
    # 学習時の標準的なData Augmentation＋正規化（CIFAR-10の平均・分散）
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                            std=(0.2470, 0.2435, 0.2616)),    ])

    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                            std=(0.2470, 0.2435, 0.2616)),    ])

    train_set = datasets.CIFAR10(root='./data', train=True, download=True,   transform=transform_train)
    test_set  = datasets.CIFAR10(root='./data', train=False, download=True,  transform=transform_eval)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    model = make_model().to(device)
    quick_train(model, train_loader,device, epochs=epochs, lr=0.1)
    # ===== 4) Hessian/HVP用の損失関数 =====
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    return model,loss_fn,[train_loader,test_loader]


Status = Literal[
    "tracking:intermediate",      # 計算グラフ上の中間テンソル（grad_fnあり）
    "tracking:leaf_trainable",    # 学習可能な葉（nn.Parameter相当）
    "frozen:leaf",                # 勾配追跡なしの葉（最初から追跡していない/固定重み等）
    "detached:view",              # detach由来が濃厚なview（_baseあり & requires_grad=False）
    "detached:likely",            # detachの可能性が高い（requires_grad=False かつ 履歴の痕跡）
    "frozen:intermediate_weird",  # まれ：中間なのに追跡オフ（特殊ケース）
]

def classify_autograd_state(t: torch.Tensor) -> Tuple[Status, Dict[str, object]]:
    """
    1つのテンソル t について、autograd的な状態を総合判定する。
    戻り値: (状態ラベル, 付帯情報ディクショナリ)
    """
    if not isinstance(t, torch.Tensor):
        raise TypeError("t must be a torch.Tensor")

    requires_grad = bool(t.requires_grad)
    grad_fn = t.grad_fn
    has_history = grad_fn is not None
    is_leaf = bool(t.is_leaf)
    is_view = getattr(t, "_base", None) is not None  # 内部属性。view/共有ストレージの強力ヒント

    info = {
        "requires_grad": requires_grad,
        "has_history": has_history,
        "is_leaf": is_leaf,
        "is_view_like": is_view,
        "grad_fn_type": type(grad_fn).__name__ if has_history else None,
        "dtype": str(t.dtype),
        "device": str(t.device),
        "shape": tuple(t.shape),
    }

    # 追跡オン
    if requires_grad:
        if has_history:
            return "tracking:intermediate", info
        # grad_fnなしで追跡オン → 典型的には学習可能な葉（nn.Parameter）
        if is_leaf:
            return "tracking:leaf_trainable", info
        # まれにleafでないのにgrad_fnが無い状況があり得る（inplaceやカスタムop絡み）
        return "tracking:intermediate", info

    # ここから追跡オフ（requires_grad=False）
    # detach() っぽさの強ヒント：view である（_baseあり）
    if is_view:
        return "detached:view", info

    # 履歴がある/あった痕跡（通常はrequires_grad Falseならgrad_fnは消えるが、
    # 特殊ケースや直後の状態観察で検出されることがある）
    if has_history and not requires_grad:
        return "detached:likely", info

    # 追跡オフの葉：最初から追跡していない定数/固定テンソル/データローダ出力など
    if is_leaf and not has_history:
        return "frozen:leaf", info

    # 追跡オフの中間（レアケース）。実用上はdetach済み/切断済みとして扱ってよいことが多い
    return "frozen:intermediate_weird", info


# 利便用: 「detachされているとみなすか？」のブール関数
def is_detached(t: torch.Tensor, strict: bool = False) -> bool:
    """
    strict=True: 'detached:view' のみ True（最も確度が高い）
    strict=False: 'detached:view' または 'detached:likely'、さらに
                  requires_grad=False かつ has_history==False でも
                  中間なら切断扱い（保守的にTrue）にする
    """
    status, info = classify_autograd_state(t)
    if strict:
        return status == "detached:view"
    if status in ("detached:view", "detached:likely"):
        return True
    # requires_gradがFalse かつ 中間っぽい（leafでない）なら切断扱い
    if not info["requires_grad"] and (not info["is_leaf"]):
        return True
    return False
