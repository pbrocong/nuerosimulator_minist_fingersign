import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class SignMNISTDataset(Dataset):
    """CSV → (이미지, 라벨) 변환. J(9), Z(25) 제외 규칙 반영."""
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self): return len(self.data_frame)

    def __getitem__(self, idx):
        orig_label = int(self.data_frame.iloc[idx, 0])
        adj_label = orig_label - 1 if orig_label > 8 else orig_label
        label = torch.tensor(adj_label, dtype=torch.long)

        pixels = self.data_frame.iloc[idx, 1:].values.astype("uint8").reshape(28,28)
        img = Image.fromarray(pixels)
        if self.transform: img = self.transform(img)
        return img, label