import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from utils import TourDataset
from utils import Preprocess
from matrix import Matrix
from NGCF import NGCF
from bprloss import BPR
from experiment import Experiment
from parsers import args


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')
if torch.cuda.is_available():
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())

# argparse dosen't support boolean type
save_data = True if args.save_data == 'True' else False
save_data = True if args.save_data == 'True' else False

FOLDER_PATH ='saved_model_data'
if not os.path.exists(FOLDER_PATH):
    os.mkdir(FOLDER_PATH)

#root_dir = '../../../LIG/Preprocessing/Datasets_v5.0/'
root_dir = '../data/'
rating_col = args.rating_col
preprocess = Preprocess(root_dir=root_dir, train_by_destination=False, folder_path=FOLDER_PATH, rating_col=rating_col, scaler=args.scaler, save_data=save_data)
total_df, train_df, test_df, num_dict = preprocess.split_train_test()

train_dataset = TourDataset(df=train_df,
                            total_df=total_df,
                            train=True,
                            rating_col=rating_col)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True,
                          drop_last=True)

test_dataset = TourDataset(df=test_df,
                           total_df=total_df,
                           train=False,
                           rating_col=rating_col)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=args.test_batch,
                         shuffle=False,
                         drop_last=True)

matrix_generator = Matrix(total_df=total_df,
                          cols=['year', 'userid', 'itemid', rating_col],
                          rating_col=rating_col,
                          num_dict=num_dict,
                          folder_path=FOLDER_PATH,
                          save_data=save_data,
                          device=device).to(device)
lap_list = matrix_generator.create_matrix()


model = NGCF(embed_size=args.embed_size,
             layer_size=[65, 65],
             node_dropout=args.node_dropout,
             mess_dropout=args.mess_dropout,
             emb_ratio=args.emb_ratio,
             lap_list=lap_list,
             num_dict=num_dict,
             batch_size=args.batch_size,
             device=device).to(device=device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = BPR(weight_decay=0.025, batch_size=args.batch_size).to(device)
test_criterion = BPR(weight_decay=0.025, batch_size=args.test_batch).to(device)

d1 = datetime.now()
train = Experiment(model=model,
                   optimizer=optimizer,
                   criterion=criterion,
                   test_criterion=test_criterion,
                   train_dataloader=train_loader,
                   test_dataloader=test_loader,
                   epochs=args.epoch,
                   ks=args.ks,
                   device=device).to(device)
train.train()
print(f'Train ended! Total Run time:{datetime.now()-d1}')

if save_data:
    d1 = datetime.now()
    PATH = os.path.join(FOLDER_PATH, f'NGCF_implicit_{args.epoch}_{args.batch_size}_{args.lr}_{args.emb_ratio}_{args.scaler}_{d1.month}_{d1.day}_{d1.hour}_{d1.minute}' + '.pth')
    torch.save(model.state_dict(), PATH)
    print('Model saved!')
