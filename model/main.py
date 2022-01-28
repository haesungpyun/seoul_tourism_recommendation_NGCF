import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import TourDataset
from utils import split_train_test
from matrix import Matrix
from NGCF import NGCF
from bprloss import BPR
from experiment import Train, Test
from parsers import args


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

root_path = '../dataset'
total_df, train_df, test_df = split_train_test(root_dir=root_path, train_by_destination=False)


num_dict = {'user': total_df['userid'].nunique(),
            'item': total_df['itemid'].nunique(),
            'day': total_df['dayofweek'].nunique(),
            'sex': total_df['sex'].nunique(),
            'age': total_df['age'].nunique(),
            'date': total_df['month-day'].nunique()}

train_dataset = TourDataset(df=train_df,
                            total_df=total_df,
                            train=True)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=args.batch_size,
                          shuffle=False,
                          drop_last=True)

test_dataset = TourDataset(df=train_df,
                            total_df=total_df,
                            train=False)

test_loader = DataLoader(dataset=test_dataset,
                          batch_size=args.test_batch,
                          shuffle=False,
                          drop_last=True)

matrix_generator = Matrix(total_df=total_df,
                        cols=['year', 'userid', 'itemid'],
                        device=device)
lap_list = matrix_generator.create_matrix()

model = NGCF(embed_size=args.embed_size,
             layer_size=[64, 64, 64],
             node_dropout=args.node_dropout,
             mess_dropout=args.mess_dropout,
             mlp_ratio=args.mlp_ratio,
             lap_list=lap_list,
             num_dict=num_dict,
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
