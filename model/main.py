import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os
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

root_dir = '../data'
preprocess = Preprocess(root_dir=root_dir, train_by_destination=False)
total_df, train_df, test_df = preprocess.split_train_test()

rating = 'visitor'
num_dict = {'user': total_df['userid'].nunique(),
            'item': total_df['itemid'].nunique(),
            'sex': total_df['sex'].max() + 1,
            'age': total_df['age'].max() + 1,
            'date': total_df['dateid'].max() + 1,
            'dayofweek': total_df['dayofweek'].max()+1}

train_dataset = TourDataset(df=train_df,
                            total_df=total_df,
                            train=True,
                            rating=rating)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=args.batch_size,
                          shuffle=False,
                          drop_last=True)

test_dataset = TourDataset(df=test_df,
                           total_df=total_df,
                           train=False,
                           rating=rating)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=args.test_batch,
                         shuffle=False,
                         drop_last=True)

matrix_generator = Matrix(total_df=total_df,
                          cols=['year', 'userid', 'itemid', rating],
                          rating=rating,
                          num_dict=num_dict,
                          device=device)

lap_list = matrix_generator.create_matrix()

model = NGCF(embed_size=args.embed_size,
             layer_size=[64, 64, 64],
             node_dropout=args.node_dropout,
             mess_dropout=args.mess_dropout,
             mlp_ratio=args.mlp_ratio,
             lap_list=lap_list,
             num_dict=num_dict,
             batch_size=args.batch_size,
             device=device).to(device=device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = BPR(weight_decay=0.025, batch_size=args.batch_size)

train = Experiment(model=model,
                   optimizer=optimizer,
                   criterion=criterion,
                   train_dataloader=train_loader,
                   test_dataloader=test_loader,
                   epochs=args.epoch,
                   ks=args.ks,
                   device=device)
train.train()
print('train ended')

model_dir = os.path.join('./', 'NGCG.pth')
torch.save(model, model_dir)

print('---------------------------------------------------------------------------------------------')
print('------------------------------------------HELP-----------------------------------------------')
print('월일 : 01 01 ~ 12 31')
print('성별 : f / m')
print('연령 : 5-9세이하 / 15-10~19 / 25-20~29 / 35-30~39 / 45-40~49 / 55-50~59 / 65-60~69 / 75-70세이상')
print('---------------------------------------------------------------------------------------------')
user_dict = preprocess.user_dict
item_dict = preprocess.item_dict
date_dict = preprocess.date_dict
print(user_dict)
print(item_dict)
print(date_dict)


dates = input("관광할 월-일을 입력하세요(ex 01 01):").split()
dow = input("관광할 요일을 입력하세요(ex mon):")
sex = input('관광객의 성별을 입력하세요(ex m):')
age = input('관광객의 연령을 입력하세요(ex 25):')


week = ['mon', 'tue', 'wed', 'thur', 'fri', 'sat', 'sun']
gender = ['f', 'm']
if dates[0][0] =='1':
    date = dates[0] + dates[1]
else:
    date = dates[0][1] + dates[1]
sex = str(gender.index(sex))
dow = week.index(dow)
u_feats = age + sex + date

u_id = user_dict[u_feats]
date = date_dict[int(date)]
u_id = torch.LongTensor([u_id])
date = torch.LongTensor([date])
age = torch.LongTensor([int(age)])
sex = torch.LongTensor([int(sex)])
dow = torch.LongTensor([dow])
u_embeds, _, _ = model(year=torch.LongTensor([0]),
                u_id=u_id,
                age=age,
                date=date,
                sex=sex,
                dow=dow,
                pos_item=torch.LongTensor([0]),
                neg_item=torch.empty(0),
                node_flag=False)

all_u_emb, all_i_emb = model.all_users_emb, model.all_items_emb
all_pred_ratings = torch.mm(u_embeds, all_i_emb.T)
_, all_rank = torch.topk(all_pred_ratings[0], 100)
recommend_des = []

for i in range(100):
    recommend_des.append(list(item_dict.keys())[list(item_dict.values()).index(all_rank[i])])

print(recommend_des)

