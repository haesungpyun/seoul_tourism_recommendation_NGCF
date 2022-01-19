import torch as t
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import TourDataset
from matrix import Matrix
from NGCF import NGCF
from bprloss import BPR
from experiment import Train, Test
from parsers import args


device = t.device('cuda' if t.cuda.is_available() else 'cpu')
print(f'device: {device}')

root_path = '../dataset'
dataset = TourDataset(root_dir=root_path,
                      label_col='congestion_1')
dates, users, items, labels =dataset.dates, dataset.users, dataset.items, dataset.labels


train_datalodaer = DataLoader(dataset=dataset,
                              batch_size=256,
                              shuffle=False,
                              drop_last=True)

n_user = len(users.unique())
n_item = len(items.unique())

print(n_item)




sparse_lap_mat, eye_mat = Matrix(users=users, items=items, device=device).create_matrix()

model = NGCF(n_user=n_user,
             n_item=n_item,
             embed_size=64,
             layer_size=[64, 64, 64],
             node_dropout=0.2,
             mess_dropout=[0.1, 0.1, 0.1],
             lap_mat=sparse_lap_mat,
             eye_mat=eye_mat,
             device=device).to(device=device)

if __name__ == '__main__':

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = BPR(weight_decay=0.025, batch_size=args.batch_size)

    train = Train(model=model,
                  optimizer=optimizer,
                  criterion=criterion,
                  train_dataloader=train_loader,
                  test_dataloader=test_loader,
                  epochs=args.epoch,
                  device=device)
    train.train()
    print('train ended')

    test = Test(model=model,
                dataloader=test_loader,
                ks=args.ks,
                device=device)









