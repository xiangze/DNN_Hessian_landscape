import os
import random, inspect
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms.functional import InterpolationMode

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

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

def simple_train(device,epochs=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    return model,loss_fn,[train_loader,test_loader]


# 画像正規化（for ImageNet）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def _build_transforms(is_train: bool, grayscale_to_rgb: bool) -> transforms.Compose:
    """ResNet50 用の前処理を返す。MNIST系は 1ch→3ch に変換。"""
    ops = []
    if grayscale_to_rgb:
        # PIL Image(L) → 3ch
        ops.append(transforms.Grayscale(num_output_channels=3))
    if is_train:
        # 小さな画像(CIFARなど)にも効くよう RandomResizedCrop を採用
        ops += [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        ops += [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)

def _make_dataset(name: str, root: str, train: bool, download: bool, tfm: transforms.Compose ) -> Tuple[torch.utils.data.Dataset, int]:
    """データセットを作り、(dataset, num_classes) を返す。"""
    n = name.lower()
    if n in {"mnist", "fashionmnist", "kmnist"}:
        cls = {
            "mnist": datasets.MNIST,
            "fashionmnist": datasets.FashionMNIST,
            "kmnist": datasets.KMNIST,
        }[n]
        ds = cls(root=root, train=train, transform=tfm, download=download)
        num_classes = 10
        return ds, num_classes
    elif n in {"cifar10", "cifar"}:
        ds = datasets.CIFAR10(root=root, train=train, transform=tfm, download=download)
        num_classes = 10
        return ds, num_classes
    elif n == "cifar100":
        ds = datasets.CIFAR100(root=root, train=train, transform=tfm, download=download)
        num_classes = 100
        return ds, num_classes
    elif n == "svhn":
        split = "train" if train else "test"
        ds = datasets.SVHN(root=root, split=split, transform=tfm, download=download)
        num_classes = 10
        return ds, num_classes
    elif n == "imagefolder":
        # 期待ディレクトリ構成:
        # root/train/<class>/*, root/val/<class>/*
        sub = "train" if train else "val"
        path = os.path.join(root, sub)
        ds = datasets.ImageFolder(path, transform=tfm)
        num_classes = len(ds.classes)
        return ds, num_classes
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Use one of: "
            "MNIST, FashionMNIST, KMNIST, CIFAR10, CIFAR100, SVHN, ImageFolder"
        )

def _seed_all(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def train_resnet50(
    dataset_name: str,
    data_root: str = "./data",
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    pretrained: bool = True,
    num_workers: int = 4,
    device: Optional[str] = None,
    download: bool = True,
    custom_num_classes: Optional[int] = None,
    use_amp: bool = True,
    seed: int = 42,
) -> dict:
    """
    ResNet50 を使って指定データセットを学習する汎用関数。
    対応: MNIST / FashionMNIST / KMNIST / CIFAR10 / CIFAR100 / SVHN / ImageFolder

    Args:
        dataset_name: データセット名（大文字小文字無視）。自前なら "ImageFolder"
        data_root: データ保存/検索ディレクトリ（ImageFolder のときはそのルート）
        epochs: エポック数
        batch_size: バッチサイズ
        lr: 学習率（SGD）
        weight_decay: L2 正則化
        momentum: SGD モーメント
        pretrained: True で ImageNet 事前学習重み
        num_workers: DataLoader の workers
        device: "cuda" / "cpu" / None(自動)
        download: True で自動ダウンロード（ImageFolder には無関係）
        custom_num_classes: クラス数を明示したい場合に指定（ImageFolder など）
        use_amp: 自動混合精度 (AMP) を使うか
        seed: 乱数シード

    Returns:
        dict: {"best_val_acc": float, "last_val_acc": float, "model": nn.Module}
    """
    _seed_all(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    name_lower = dataset_name.lower()
    grayscale = name_lower in {"mnist", "fashionmnist", "kmnist"}
    tfm_train = _build_transforms(is_train=True,  grayscale_to_rgb=grayscale)
    tfm_eval  = _build_transforms(is_train=False, grayscale_to_rgb=grayscale)

    # Dataset & Loader
    train_ds, num_classes_train = _make_dataset(name_lower, data_root, train=True,  download=download, tfm=tfm_train)
    val_ds,   num_classes_val   = _make_dataset(name_lower, data_root, train=False, download=download, tfm=tfm_eval)

    num_classes = custom_num_classes if custom_num_classes is not None else max(num_classes_train, num_classes_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Model
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    model.to(device)

    # Optimizer & Scheduler & Loss
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    # CosineAnnealing でゆるやかに学習率を下げる（お好みで変更可）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_acc = 0.0
    last_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        total = 0

        for imgs, targets in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(imgs)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = imgs.size(0)
            total += bs
            running_loss += loss.item() * bs
            running_acc  += _top1_accuracy(logits.detach(), targets) * bs

        train_loss = running_loss / total
        train_acc  = running_acc / total

        # ---- eval ----
        model.eval()
        val_total = 0
        val_running_acc = 0.0
        val_running_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(imgs)
                    loss = criterion(logits, targets)
                bs = imgs.size(0)
                val_total += bs
                val_running_loss += loss.item() * bs
                val_running_acc  += _top1_accuracy(logits, targets) * bs

        last_val_acc = val_running_acc / val_total
        val_loss = val_running_loss / val_total
        scheduler.step()

        print(f"[{epoch:03d}/{epochs}] "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={last_val_acc:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.6f}")

        if last_val_acc > best_val_acc:
            best_val_acc = last_val_acc

    return {"best_val_acc": best_val_acc, "last_val_acc": last_val_acc, "model": model}


# -------- モデル切り替え周り --------
def _first_non_none(*vals):
    for v in vals:
        if v is not None: return v
    return None

def _build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    """
    torchvision の主要な分類モデルを名称で生成し、分類ヘッドを num_classes に付け替える。
    """
    name = model_name.lower()

    # 動的に“存在するものだけ”登録（古い torchvision でも安全）
    CANDIDATES = {}
    def add(key, fn_name, weights_enum_name):
        if hasattr(models, fn_name):
            fn = getattr(models, fn_name)
            weights_enum = getattr(models, weights_enum_name, None)
            CANDIDATES[key] = (fn, weights_enum)

    add("resnet18", "resnet18", "ResNet18_Weights")
    add("resnet34", "resnet34", "ResNet34_Weights")
    add("resnet50", "resnet50", "ResNet50_Weights")
    add("resnet101", "resnet101", "ResNet101_Weights")
    add("resnext50_32x4d", "resnext50_32x4d", "ResNeXt50_32X4D_Weights")
    add("wide_resnet50_2", "wide_resnet50_2", "Wide_ResNet50_2_Weights")
    add("mobilenet_v3_small", "mobilenet_v3_small", "MobileNet_V3_Small_Weights")
    add("mobilenet_v3_large", "mobilenet_v3_large", "MobileNet_V3_Large_Weights")
    add("efficientnet_b0", "efficientnet_b0", "EfficientNet_B0_Weights")
    add("efficientnet_b1", "efficientnet_b1", "EfficientNet_B1_Weights")
    add("efficientnet_b2", "efficientnet_b2", "EfficientNet_B2_Weights")
    add("efficientnet_b3", "efficientnet_b3", "EfficientNet_B3_Weights")
    add("convnext_tiny", "convnext_tiny", "ConvNeXt_Tiny_Weights")
    add("convnext_small", "convnext_small", "ConvNeXt_Small_Weights")
    add("vit_b_16", "vit_b_16", "ViT_B_16_Weights")
    add("swin_t_v2", "swin_t_v2", "Swin_T_V2_Weights")

    if name not in CANDIDATES:
        available = ", ".join(sorted(CANDIDATES.keys()))
        raise ValueError(f"Unknown/unsupported model '{model_name}'. Available: {available}")

    fn, weights_enum = CANDIDATES[name]

    # weights 引数が使える場合は DEFAULT→V2→V1 の順で選択。古いAPIなら pretrained=bool を使う
    weights = None
    if pretrained and weights_enum is not None:
        weights = _first_non_none(
            getattr(weights_enum, "DEFAULT", None),
            getattr(weights_enum, "IMAGENET1K_V2", None),
            getattr(weights_enum, "IMAGENET1K_V1", None),
        )

    try:
        sig = inspect.signature(fn)
        if "weights" in sig.parameters:
            model = fn(weights=weights)
        elif "pretrained" in sig.parameters:
            model = fn(pretrained=pretrained)
        else:
            model = fn()
    except Exception:
        model = fn(pretrained=pretrained) if pretrained else fn()

    # 分類ヘッドの付け替え（モデルごとに場所が違う）
    def _replace_linear(module: nn.Module, attr_path: str):
        # attr_path 例: "fc", "classifier.3", "heads.head", "classifier.2", "head"
        obj = module
        parts = attr_path.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        last = parts[-1]
        old: nn.Module = getattr(obj, last)
        in_features = old.in_features if isinstance(old, nn.Linear) else None
        if in_features is None:
            # MobileNet/EfficientNet/ConvNeXt など Sequential の末尾を探す
            if hasattr(obj, last) and isinstance(old, nn.Sequential) and len(old) > 0:
                # 末尾の Linear を探す
                for i in range(len(old)-1, -1, -1):
                    if isinstance(old[i], nn.Linear):
                        in_features = old[i].in_features
                        old[i] = nn.Linear(in_features, num_classes)
                        setattr(obj, last, old)
                        return
            raise RuntimeError(f"Could not find Linear layer at '{attr_path}' for {model_name}")
        setattr(obj, last, nn.Linear(in_features, num_classes))

    if name.startswith("resnet") or name.startswith("resnext") or name.startswith("wide_resnet"):
        _replace_linear(model, "fc")
    elif name.startswith("mobilenet"):
        # classifier[-1] が Linear
        _replace_linear(model, "classifier")
    elif name.startswith("efficientnet"):
        _replace_linear(model, "classifier")
    elif name.startswith("convnext"):
        _replace_linear(model, "classifier")
    elif name.startswith("vit"):
        # torchvision ViT は heads.head が最終線形
        if hasattr(model, "heads") and hasattr(model.heads, "head"):
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)
        else:
            _replace_linear(model, "heads")
    elif name.startswith("swin"):
        # Swin は head
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Classifier replacement not implemented for '{model_name}'")

    return model

def _top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def train_image_classifier(
    model_name: str = "resnet50",
    dataset_name: str = "CIFAR10",
    data_root: str = "./data",
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    pretrained: bool = True,
    num_workers: int = 4,
    device: Optional[str] = None,
    download: bool = True,
    custom_num_classes: Optional[int] = None,
    use_amp: bool = True,
    seed: int = 42,
    freeze_backbone: bool = False,
) -> dict:
    """
    任意モデル(ResNet/ResNeXt/WideResNet/MobileNetV3/EfficientNet/ConvNeXt/ViT/Swin)と
    任意データセット(MNIST系/CIFAR/SVHN/ImageFolder)を切り替えて学習。

    画像は 224x224・ImageNet正規化。MNIST系は自動で 1ch→3ch 変換。

    Returns:
        {"best_val_acc": float, "last_val_acc": float, "model": nn.Module}
    """
    _seed_all(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    name_lower = dataset_name.lower()
    grayscale = name_lower in {"mnist", "fashionmnist", "kmnist"}
    tfm_train = _build_transforms(is_train=True,  grayscale_to_rgb=grayscale)
    tfm_eval  = _build_transforms(is_train=False, grayscale_to_rgb=grayscale)

    train_ds, n_tr = _make_dataset(name_lower, data_root, train=True,  download=download, tfm=tfm_train)
    val_ds,   n_va = _make_dataset(name_lower, data_root, train=False, download=download, tfm=tfm_eval)
    num_classes = custom_num_classes if custom_num_classes is not None else max(n_tr, n_va)

    pin = (device == "cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    model = _build_model(model_name, num_classes=num_classes, pretrained=pretrained).to(device)
    if freeze_backbone:
        # 最終分類器以外を凍結（勾配停止）
        for n, p in model.named_parameters():
            p.requires_grad = ("fc" in n) or ("head" in n) or ("classifier" in n) or ("heads" in n)
    model = model.to(memory_format=torch.channels_last)  # 速度最適化（対応モデルのみ恩恵）

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_acc, last_val_acc = 0.0, 0.0
    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        tot, run_loss, run_acc = 0, 0.0, 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x); loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()

            bs = x.size(0); tot += bs
            run_loss += loss.item() * bs
            run_acc  += _top1_accuracy(logits.detach(), y) * bs
        train_loss, train_acc = run_loss / tot, run_acc / tot

    ### ---- eval ----
        model.eval()
        vtot, vacc, vloss = 0, 0.0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(x); loss = criterion(logits, y)
                bs = x.size(0); vtot += bs
                vloss += loss.item() * bs
                vacc  += _top1_accuracy(logits, y) * bs
        last_val_acc = vacc / vtot; vloss = vloss / vtot
        scheduler.step()

        print(f"[{epoch:03d}/{epochs}] "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={vloss:.4f} acc={last_val_acc:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.6f}")

        best_val_acc = max(best_val_acc, last_val_acc)

    return {"best_val_acc": best_val_acc, "last_val_acc": last_val_acc, "model": model,
            "dataloaders":[train_loader,val_loader],
            "loss_fn":nn.CrossEntropyLoss(reduction='mean')}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='色々なdatasetを色々なnetworkで学習数r）') 
    parser.add_argument('--model', default='resnet50')
    parser.add_argument('--dataset', default='MNIST')
    parser.add_argument('--batchsize', default=128,type=int)
    parser.add_argument('--lr', default=0.05,type=float)
    parser.add_argument('--epochs', default=5,type=int)
    args = parser.parse_args()
    model_name=args.model
    dataset_name=args.dataset
    batchsize=args.batchsize
    lr=args.lr
    epochs=args.epochs
    train_image_classifier(model_name=model_name, dataset_name=dataset_name, epochs=epochs, batch_size=batchsize, lr=lr)
    #train_image_classifier(model_name="resnet50", dataset_name="CIFAR10", epochs=2, batch_size=256, lr=0.05)
    # 例: MNIST × MobileNetV3（1ch→3chは自動）
    #train_image_classifier(model_name="mobilenet_v3_small", dataset_name="MNIST", epochs=2, batch_size=256, lr=0.05)
    #train_image_classifier(model_name="efficientnet_b0", dataset_name="CIFAR100", epochs=2, batch_size=128, lr=0.05)
    #train_image_classifier(model_name="convnext_tiny", dataset_name="ImageFolder", data_root="/path/to/ds")


