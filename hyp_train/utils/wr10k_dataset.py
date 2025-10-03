from torch.utils.data import Dataset
from PIL import Image


class WR10kDataset(Dataset):
    def __init__(self, df, root, transform, id_list=None):
        self.df = df.reset_index(drop=True)
        ids = sorted(id_list if id_list is not None else self.df['identity'].unique().tolist())
        self.id2idx = {k: i for i, k in enumerate(ids)}
        self.root = root
        self.tfm = transform
        self.transform = self.tfm
        self.num_classes = len(self.id2idx)
        self.id_list = ids

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(f"{self.root}/{r['path']}").convert("RGB")
        x = self.tfm(img)
        y = self.id2idx[r['identity']]
        return x, y, r['path'], r['identity']