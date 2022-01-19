import torch as t
from torch.utils.data import Dataset
import pandas as pd
import os


class TourDataset(Dataset):
    def __init__(self, root_dir, label_col='congestion_1'):
        super(TourDataset, self).__init__()

        self.path = os.path.join(root_dir, 'scale_data.csv')
        self.df = pd.read_csv(self.path)
        self.label_col = label_col

        self.dates, self.users, self.items, self.labels = self._split_data_labels()

    def __len__(self):
        return len(self.user_features)

    def __getitem__(self, idx):
        return self.dates[idx], self.user_features[idx], self.items[idx], self.labels[idx]


    def _split_data_labels(self):
        df = self.df
        user_col_list = [col for col in df.columns if col not in ['date', 'destination', 'congestion_1', 'congestion_2']]
        dates, users, items, labels = df['date'], df[user_col_list], df['destination'], df[self.label_col]

        dates = dates.values.tolist()
        users = users.values.tolist()
        items = items.values.tolist()
        labels = labels.values.tolist()

        print(f'len(dates):{len(dates)}, len(users):{len(users)}, len(items):{len(items)}, len(labels):{len(labels)}')

        return t.FloatTensor(dates), t.FloatTensor(users), t.LongTensor(items), t.FloatTensor(labels)









