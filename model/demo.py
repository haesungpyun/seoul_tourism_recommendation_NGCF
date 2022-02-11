import os
import numpy as np
import pandas as pd
import torch
import pickle
from NGCF import NGCF
from parsers import args


def input_filterchar(userinfo: str):
    str = ""
    for token in userinfo:
        if ord(token) < 48 or ord(token) > 57:
            break
        str += token
    return int(str)


if __name__ == '__main__':
    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # print GPU information
    if torch.cuda.is_available():
        print('Current cuda device:', torch.cuda.current_device())
        print('Count of using GPUs:', torch.cuda.device_count())

    FOLDER_PATH = 'saved_model_data'

    print('---------------------Load Id Data---------------------')
    PATH = os.path.join(FOLDER_PATH, f'user_dict' + '.pkl')
    with open(PATH, 'rb') as f:
        user_dict = pickle.load(f)
    PATH = os.path.join(FOLDER_PATH, f'item_dict' + '.pkl')
    with open(PATH, 'rb') as f:
        item_dict = pickle.load(f)
    PATH = os.path.join(FOLDER_PATH, f'num_dict' + '.pkl')
    with open(PATH, 'rb') as f:
        num_dict = pickle.load(f)
        print(num_dict)
    print('User Id, Item Id, Number Data Loaded!')

    print('---------------------Load Lapliacian Data---------------------')
    PATH = os.path.join(FOLDER_PATH, f'lap_list' + '.pkl')
    with open(PATH, 'rb') as f:
        lap_list = pickle.load(f)
    print('Laplacian Matrix Data Loaded!')

    print('---------------------Load Model---------------------')
    model = NGCF(embed_size=args.embed_size,
                 layer_size=[64, 64, 64],
                 node_dropout=args.node_dropout,
                 mess_dropout=args.mess_dropout,
                 mlp_ratio=args.mlp_ratio,
                 lap_list=lap_list,
                 num_dict=num_dict,
                 batch_size=args.batch_size,
                 device=device).to(device=device)
    PATH = os.path.join(FOLDER_PATH, f'NGCF_dow_0.5_visitor_9' + '.pth')
    model.load_state_dict(torch.load(PATH))
    model.eval()
    print('NGCF Model Loaded!')

    print('---------------------Load Destination Data---------------------')
    # root_dir = '../../../LIG/Preprocessing/Datasets_v5.0/'
    root_dir = '../data/'
    path = os.path.join(root_dir, 'destination_id_name.csv')
    df_id_name = pd.read_csv(path)
    df_id_name = df_id_name.sort_values(by='destination').reset_index().drop('index', axis=1)
    print("Destination Data Loaded!")

    num_list = ['첫', '두', '세', '네']
    week = ['월', '화', '수', '목', '금', '토', '일']
    gender = ['여', '남']
    month_info = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

    print('---------------------------------------------------------------------------------------------')
    print(user_dict.keys())
    num = input("관광객 수를 입력하세요(ex 2):")
    duration = input("관광 기간를 입력하세요(ex 7):")
    dates = input("관광할 시작 월-일-요일을 입력하세요(ex 01 01 수):").split()
    rec_num = input("추천 받을 관광지의 개수를 입력하세요(ex 10)")

    month = torch.LongTensor([int(dates[0])]).to(device)
    day = torch.LongTensor([int(dates[1])]).to(device)
    dow = torch.LongTensor([week.index(dates[2])]).to(device)

    total_user_info = []
    for i in range(int(num)):
        sex = input(f'{num_list[i]}번째 관광객의 성별을 입력하세요(ex 남/여):')
        age = input(f'{num_list[i]}번째 관광객의 연령을 입력하세요(ex 23):')

        sex = str(gender.index(sex))
        age = str((int(age) // 10) * 10 + 5)
        u_feats = age + sex + str(month.item()) + str(day.item())
        uid = user_dict[u_feats]
        uid = torch.LongTensor([uid])
        age = torch.LongTensor([int(age)])
        sex = torch.LongTensor([int(sex)])

        for length in range(int(duration)):
            user_info = [uid, age, sex]
            day_tmp = day + length
            dow_tmp = dow + length
            dow_tmp = dow_tmp % 7
            if day > month_info[int(dates[0])]:
                month += 1
            user_info += [month, day_tmp, dow_tmp]
            total_user_info.append(user_info)

    print('-----------------------------추천 관광지 산출 중...-----------------------------')
    total_user_info = torch.LongTensor(total_user_info)
    print(total_user_info)
    # user_info = [u_id, age, sex, month, day, dow]
    u_id, age, sex = total_user_info.T[0], total_user_info.T[1], total_user_info.T[2]
    month, day, dow = total_user_info.T[3], total_user_info.T[4], total_user_info.T[5]

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
    all_rating, all_rank = torch.topk(all_pred_ratings, 100)

    print(all_rating, all_rank)
    all_zip = zip(all_rating, all_rank)
    # 11 29 1 25 / 11 29 0 65 / 11 29 0 15
    recommend_des = []
    n = 1
    d = 0
    df_tmp = df_id_name
    for i in range(int(duration) * int(num)):
        if d < int(duration):
            d += 1
        else:
            n += 1
            d = 1

        user_df = df_tmp.iloc[all_rank[i].tolist()]
        day_vis = user_df.copy()
        user_df.loc[:, 'rating'] = all_rating[i].detach().numpy()
        day_vis.loc[:, str(d)] = 0
        day_vis.loc[:, str(d)] += all_rating[i].detach().numpy()

        print(f'--------------{n}번째 관광객의 {d}일째 추천 여행지 입니다.--------------')
        print(user_df.iloc[:int(rec_num)].reset_index().drop('index', axis=1))


