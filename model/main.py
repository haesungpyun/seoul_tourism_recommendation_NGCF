import torch as t
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import TourDataset
from utils import split_train_test
from matrix import Matrix
from NGCF import NGCF
from bprloss import BPR
from experiment import Train, Test
from parsers import args


device = t.device('cuda' if t.cuda.is_available() else 'cpu')
print(f'device: {device}')

root_path = '../dataset'
total_df, train_df, test_df = split_train_test(root_dir=root_path, label_col='congestion_1')

"""
train_dataset = TourDataset(df=train_df,
                            total_df=total_df,
                            label_col='congestion_1',
                            train=True)
"""
test_dataset = TourDataset(df=train_df,
                            total_df=total_df,
                            label_col='congestion_1',
                            train=False)

for i, u in test_dataset:
    print(i)
    print(u)
    print(i.shape)
    print(u.shape)
    break

"""
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=256,
                          shuffle=False,
                          drop_last=True)
"""
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=256,
                          shuffle=False,
                          drop_last=True)

for u,p in test_loader:
    print(u)
    print(p)
    print(u.shape)
    print(p.shape)
    break


# train 코드에 dataloader 부분에 넣기
matrix_generator = Matrix(total_df=total_df,
                        cols=['age', 'dayofweek', 'time', 'sex'],
                        label_col='congestion_1',
                        device=device)
lap_list, eye_mat = matrix_generator.create_matrix()

model = NGCF(n_user=n_user,
             n_item=n_item,
             embed_size=64,
             layer_size=[64, 64, 64],
             node_dropout=0.2,
             mess_dropout=[0.1, 0.1, 0.1],
             mlp_ratio=0.5,
             lap_list=lap_list,
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









