import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_accuracy(train_hist, test_hist, title="Accuracy"):
    plt.figure(figsize=(8,4))
    plt.plot(train_hist, marker="o", label="Train")
    plt.plot(test_hist, marker="o", label="Test")
    plt.title(title); plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
    plt.grid(True); plt.legend(); plt.ylim(0,101); plt.tight_layout(); plt.show()

def plot_weight_distribution(model, fitter, title_suffix=""):
    import numpy as np
    import matplotlib.pyplot as plt
    layers = [("fc1", getattr(model, "fc1", None)),
              ("fc2", getattr(model, "fc2", None))]
    layers = [(n,l) for n,l in layers if l is not None]
    fig, axes = plt.subplots(1, len(layers), figsize=(6*len(layers),4))
    if len(layers)==1: axes = [axes]
    for ax,(name,layer) in zip(axes, layers):
        w = layer.weight.data.cpu().numpy().ravel()
        w_unscaled = fitter.unscale(w)
        ax.hist(w_unscaled, bins=50, alpha=0.75, edgecolor="black")
        ax.set_title(f"{name} weights {title_suffix}")
        ax.set_xlabel("Conductance"); ax.set_ylabel("Count")
        ax.grid(True, ls="--", alpha=0.5)
    plt.tight_layout(); plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9,7))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix"); plt.xlabel("Pred"); plt.ylabel("True")
    plt.tight_layout(); plt.show()

def plot_specific_letter_accuracy(history_dict):
    import matplotlib.pyplot as plt
    epochs = range(1, len(next(iter(history_dict.values())))+1)
    plt.figure(figsize=(9,5))
    for ch, acc in history_dict.items():
        plt.plot(epochs, acc, marker="o", label=f'"{ch}"')
    plt.title("Specific Letter Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
    plt.ylim(0,101); plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()