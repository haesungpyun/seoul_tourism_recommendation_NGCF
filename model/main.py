import os
import numpy as np
import pandas as pd
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
save_model = True if args.save_model == 'True' else False
save_data = True if args.save_data == 'True' else False

FOLDER_PATH ='saved_model_data'
if not os.path.exists(FOLDER_PATH):
    os.mkdir(FOLDER_PATH)

#root_dir = '../../../LIG/Preprocessing/Datasets_v5.0/'
root_dir = '../data/'
preprocess = Preprocess(root_dir=root_dir, train_by_destination=False, folder_path=FOLDER_PATH, save_data=save_data)
total_df, train_df, test_df = preprocess.split_train_test()

rating_col = 'visitor'
num_dict = {'user': total_df['userid'].nunique(),
            'item': total_df['itemid'].nunique(),
            'sex': total_df['sex'].max() + 1,
            'age': total_df['age'].max() + 1,
            'month': total_df['month'].max() + 1,
            'day': total_df['day'].max() + 1,
            'dayofweek': total_df['dayofweek'].max() + 1}

train_dataset = TourDataset(df=train_df,
                            total_df=total_df,
                            train=True,
                            rating_col=rating_col)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=args.batch_size,
                          shuffle=False,
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
test_criterion = BPR(weight_decay=0.025, batch_size=args.test_batch)

d1 = datetime.now()
train = Experiment(model=model,
                   optimizer=optimizer,
                   criterion=criterion,
                   test_criterion=test_criterion,
                   train_dataloader=train_loader,
                   test_dataloader=test_loader,
                   epochs=args.epoch,
                   ks=args.ks,
                   device=device)
train.train()
print(f'Train ended! Total Run time:{datetime.now()-d1}')


if save_model:
    MODEL_PATH = os.path.join(FOLDER_PATH, f'NGCF_dow_{args.mlp_ratio}_{rating_col}_{np.random.randint(10)}' + '.pth')
    torch.save(model.state_dict(), MODEL_PATH)
    print('Model saved!')

print('---------------------------------------------------------------------------------------------')
print('------------------------------------------HELP-----------------------------------------------')
print('월일 : 01 01 ~ 12 31')
print('성별 : f / m')
print('연령 : 5-9세이하 / 15-10~19 / 25-20~29 / 35-30~39 / 45-40~49 / 55-50~59 / 65-60~69 / 75-70세이상')
print('---------------------------------------------------------------------------------------------')
user_dict = preprocess.user_dict
item_dict = preprocess.item_dict
date_dict = preprocess.date_dict
print(user_dict.keys())

week = ['월', '화', '수', '목', '금', '토', '일']
gender = ['여', '남']
# root_dir = '../../../LIG/Preprocessing/Datasets_v5.0/'
root_dir = '../data/'
path = os.path.join(root_dir, 'destination_id_name.csv')
df_id_name = pd.read_csv(path)

num = input("관광객 수를 입력하세요(ex 2):")
num_list = ['첫', '두', '세', '네']

for i in range(int(num)):
    dates = input(f"{num_list[i]}번째 관광객이 관광할 월-일-요일을 입력하세요(ex 01 01 수):").split()
    sex = input(f'{num_list[i]}번째 관광객의 성별을 입력하세요(ex 남/여):')
    age = input(f'{num_list[i]}번째 관광객의 연령을 입력하세요(ex 23):')
    rec_num = input(f'{num_list[i]}번째 관광객이 추천 받을 관관지의 개수를 입력하세요(ex 10):')

    month = int(dates[0])
    day = int(dates[1])
    dow = week.index(dates[2])
    sex = str(gender.index(sex))
    age = str((int(age)//10)*10 + 5)

    u_feats = age + sex + str(month) + str(day)

    u_id = user_dict[u_feats]
    u_id = torch.LongTensor([u_id])
    month = torch.LongTensor([int(month)])
    day = torch.LongTensor([int(day)])
    age = torch.LongTensor([int(age)])
    sex = torch.LongTensor([int(sex)])
    dow = torch.LongTensor([dow])
    u_embeds, _, _ = model(year=torch.LongTensor([1]),
                           u_id=u_id,
                           age=age,
                           month=month,
                           day=day,
                           sex=sex,
                           dow=dow,
                           pos_item=torch.LongTensor([0]),
                           neg_item=torch.empty(0),
                           node_flag=False)

    all_u_emb, all_i_emb = model.all_users_emb, model.all_items_emb
    all_pred_ratings = torch.mm(u_embeds, all_i_emb.T)
    _, all_rank = torch.topk(all_pred_ratings[0], int(rec_num))

    recommend_des = []

    for i in range(int(rec_num)):
        des_id = list(item_dict.keys())[list(item_dict.values()).index(all_rank[i])]
        recommend_des.append(np.array(df_id_name.loc[df_id_name['destination'] == des_id, 'destination_name']))
    print(recommend_des)
