from torch.utils.data import Dataset
    

class PositivePairDataset(Dataset):
    def __init__(self, user_pos_dict):
        self.pairs = [(u, i) for u, items in user_pos_dict.items() for i in items]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]