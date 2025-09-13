import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

def train_epoch(model, device, loader, optimizer, epoch):
    model.train()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = nn.functional.nll_loss(out, y) if out.ndim==2 and out.shape[1]==10 else nn.CrossEntropyLoss()(out, y)
        loss.backward()
        optimizer.step()
        pred = out.argmax(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total

@torch.no_grad()
def test_epoch(model, device, loader):
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total

@torch.no_grad()
def test_specific_letters(model, device, full_test_dataset, target_chars="GCU"):
    """Sign-MNIST에서 지정한 글자들의 정확도만 계산"""
    model.eval()
    # 원 CSV의 알파벳(0~25) → 조정 라벨(0~23) 매핑 보정
    alphabet_map = {chr(65+i): i for i in range(26)}
    wanted = sorted(set(target_chars))
    target_original = [alphabet_map[c] for c in wanted]
    if "label" not in full_test_dataset.data_frame.columns:
        return {c: 0.0 for c in wanted}

    idx = full_test_dataset.data_frame[
        full_test_dataset.data_frame["label"].isin(target_original)
    ].index
    if len(idx) == 0:
        return {c: 0.0 for c in wanted}
    loader = DataLoader(Subset(full_test_dataset, idx), batch_size=128, shuffle=False)

    # 조정 라벨 → 알파벳
    adj_to_chr = {i: chr(65+i) for i in range(9)}
    adj_to_chr.update({i: chr(65+i+1) for i in range(9,24)})

    corr = {c:0 for c in wanted}
    tot  = {c:0 for c in wanted}

    for x,y in loader:
        x,y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        for yi, pi in zip(y.tolist(), pred.tolist()):
            if yi in adj_to_chr:
                ch = adj_to_chr[yi]
                if ch in tot:
                    tot[ch] += 1
                    if yi == pi: corr[ch] += 1
    return {c: (100*corr[c]/tot[c]) if tot[c]>0 else 0.0 for c in wanted}