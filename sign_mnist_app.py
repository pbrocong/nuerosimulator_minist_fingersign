import sys
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from neurosim.io import read_split_by_write_vh
from neurosim.fitter import NeuroSimFitter
from neurosim.optimizer import NeuroSimOptimizer
from neurosim.models import SimpleCNN
from neurosim.datasets import SignMNISTDataset
from neurosim.train_eval import train_epoch, test_epoch, test_specific_letters
from neurosim.plots import plot_specific_letter_accuracy
from neurosim.save_utils import save_sign_mnist_results_to_excel

def main():
    ltp_ltd_path = input("LTP/LTD 엑셀 경로: ").strip().strip("'\"")
    train_csv = input("train CSV(sign_mnist_train.csv) 경로: ").strip().strip("'\"")
    test_csv  = input("test CSV(sign_mnist_test.csv) 경로: ").strip().strip("'\"")

    ltp, ltd = read_split_by_write_vh(ltp_ltd_path)
    if ltp is None: sys.exit(1)

    TARGET_RANGE = (-1.0, 1.0)
    BATCH, EPOCHS, LR, PSF, NUM_CLASSES = 128, 15, 1e-4, 300, 24

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_ds = SignMNISTDataset(train_csv, transform=tfm)
    test_ds  = SignMNISTDataset(test_csv,  transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)

    fitter = NeuroSimFitter(ltp, ltd, target_range=TARGET_RANGE)
    model = SimpleCNN(*TARGET_RANGE, num_classes=NUM_CLASSES).to(device)
    opt = NeuroSimOptimizer(model.parameters(), lr=LR, fitter=fitter, pulse_scaling_factor=PSF)

    total_hist, letter_hist = [], {c: [] for c in "GCU"}
    for ep in range(1, EPOCHS+1):
        tr = train_epoch(model, device, train_loader, opt, ep)
        te = test_epoch(model, device, test_loader)
        total_hist.append(te)
        part = test_specific_letters(model, device, test_ds, "GCU")
        for c, a in part.items(): letter_hist[c].append(a)
        print(f"[Epoch {ep}] train={tr:.2f}% / test={te:.2f}%")

    plot_specific_letter_accuracy(letter_hist)
    save_sign_mnist_results_to_excel(total_hist, letter_hist)

if __name__ == "__main__":
    main()