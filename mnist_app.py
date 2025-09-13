import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from neurosim.io import read_split_by_write_vh
from neurosim.fitter import NeuroSimFitter
from neurosim.optimizer import NeuroSimOptimizer
from neurosim.models import SimpleNet
from neurosim.train_eval import train_epoch, test_epoch
from neurosim.plots import plot_accuracy, plot_weight_distribution
from neurosim.save_utils import save_accuracy_to_excel

def main():
    file_path = input("LTP/LTD 엑셀 경로: ").strip().strip("'\"")
    ltp, ltd = read_split_by_write_vh(file_path)
    if ltp is None: return

    TARGET_RANGE = (-1.0, 1.0)
    BATCH, EPOCHS, LR, PSF = 128, 10, 1e-4, 300

    fitter = NeuroSimFitter(ltp, ltd, target_range=TARGET_RANGE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST("./data", train=False, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)

    model = SimpleNet(*TARGET_RANGE).to(device)
    opt = NeuroSimOptimizer(model.parameters(), lr=LR, fitter=fitter, pulse_scaling_factor=PSF)

    train_hist, test_hist = [], []
    for ep in range(1, EPOCHS+1):
        tr = train_epoch(model, device, train_loader, opt, ep)
        te = test_epoch(model, device, test_loader)
        train_hist.append(tr); test_hist.append(te)
        print(f"[Epoch {ep}] train={tr:.2f}% / test={te:.2f}%")

    plot_weight_distribution(model, fitter, "Before/After")
    plot_accuracy(train_hist, test_hist, "MNIST Accuracy")
    save_accuracy_to_excel(train_hist, test_hist)

if __name__ == "__main__":
    main()