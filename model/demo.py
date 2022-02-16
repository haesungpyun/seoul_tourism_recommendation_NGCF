import os
import numpy as np
import pandas as pd
import torch
import pickle
from NGCF import NGCF
from parsers import args
import numpy as np
import io
from haversine import haversine


def input_filterchar(userinfo: str):
    str = ""
    for token in userinfo:
        if ord(token) < 48 or ord(token) > 57:
            break
        str += token
    return int(str)


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def calculate_distance(dep_x, dep_y, arr_x, arr_y):
    return np.sqrt(
        ((np.cos(dep_x) * 6400 * 2 * 3.14 / 360) * np.abs(dep_y - arr_y)) ** 2 + (111 * np.abs(dep_x - arr_x) ** 2))


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
        user_dict = CPU_Unpickler(f).load()
    PATH = os.path.join(FOLDER_PATH, f'item_dict' + '.pkl')
    with open(PATH, 'rb') as f:
        item_dict = CPU_Unpickler(f).load()
    PATH = os.path.join(FOLDER_PATH, f'num_dict' + '.pkl')
    with open(PATH, 'rb') as f:
        num_dict = CPU_Unpickler(f).load()
        print(num_dict)
    print('User Id, Item Id, Number Data Loaded!')

    print('---------------------Load Lapliacian Data---------------------')
    PATH = os.path.join(FOLDER_PATH, f'lap_list' + '.pkl')
    with open(PATH, 'rb') as f:
        lap_list = CPU_Unpickler(f).load()
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
    PATH = os.path.join(FOLDER_PATH, f'NGCF_dow_0.5_visitor_3' + '.pth')
    model.load_state_dict(torch.load(PATH, map_location=device))
    model.eval()
    print('NGCF Model Loaded!')

    print('---------------------Load Destination Data---------------------')
    # root_dir = '../../../LIG/Preprocessing/Datasets_v5.0/'
    root_dir = '../data/'
    PATH = os.path.join(root_dir, 'destination_id_name_genre_coordinate' + '.pkl')
    with open(PATH, 'rb') as f:
        df_id_name_genre_coordinate = CPU_Unpickler(f).load()
    df_id_name_genre_coordinate = df_id_name_genre_coordinate.sort_values(by='destination').reset_index().drop('index',
                                                                                                               axis=1)
    df_id_name_genre_coordinate = df_id_name_genre_coordinate.rename(columns={'middle_category_name': 'genre'})
    PATH = os.path.join(root_dir, 'seoul_gu_dong_coordinate' + '.pkl')
    with open(PATH, 'rb') as f:
        df_departure_coordinate = CPU_Unpickler(f).load()
    PATH = os.path.join(root_dir, 'congestion_1_2' + '.pkl')
    with open(PATH, 'rb') as f:
        df_congestion = CPU_Unpickler(f).load()
    df_congestion = pd.pivot_table(df_congestion, index=['month', 'day', 'dayofweek', 'destination'],
                                   aggfunc={'congestion_1': 'sum',
                                            'congestion_2': 'sum'})
    df_congestion = df_congestion.reset_index()
    print("Destination Data Loaded!")

    num_list = ['첫', '두', '세', '네']
    week = ['월', '화', '수', '목', '금', '토', '일']
    gender = ['여', '남']
    month_info = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    dest_dict = {'1': '역사관광지', '2': '휴양관광지', '3': '체험관광지', '4': '문화시설', '5': '건축/조형물', '6': '자연관광지', '7': '쇼핑'}
    rank2rate = []
    for i in range(100):
        rank2rate.append(i)

    print('---------------------------------------------------------------------------------------------')
    num = input("관광객 수를 입력하세요(ex 2):")
    duration = input("관광 기간를 입력하세요(ex 7):")
    dates = input("관광할 시작 월-일-요일을 입력하세요(ex 01 01 수):").split()
    rec_num = input("추천 받을 관광지의 개수를 입력하세요(ex 10)")
    depart = input('출발지를 입력하세요 (ex) 사직동)')

    month = torch.LongTensor([int(dates[0])]).to(device)
    day = torch.LongTensor([int(dates[1])]).to(device)
    dow = torch.LongTensor([week.index(dates[2])]).to(device)

    total_user_info = []
    for i in range(int(num)):
        sex = input(f'{num_list[i]}번째 관광객의 성별을 입력하세요(ex 남/여):')
        age = input(f'{num_list[i]}번째 관광객의 연령을 입력하세요(ex 23):')
        sex = int(gender.index(sex))
        age = ((int(age) // 10) * 10 + 5)

        day_tmp = day.item() - 1
        month_tmp = month.item()
        for length in range(int(duration)):
            dow_tmp = dow + length
            dow_tmp = dow_tmp % 7
            day_tmp = day_tmp + 1

            if day_tmp > month_info[month_tmp]:
                day_tmp = day_tmp % month_info[month_tmp]
                month_tmp += 1

            m_str = str(month_tmp)
            d_str = str(day_tmp)
            if month_tmp < 10:
                m_str = '0' + m_str
            if day_tmp < 10:
                d_str = '0' + d_str

            u_feats = str(age) + str(sex) + m_str + d_str
            uid = user_dict[u_feats]
            user_info = [uid, age, sex, month_tmp, day_tmp, dow_tmp]
            total_user_info.append(user_info)

    genre = input("관광지의 유형을 선택하세요\n"
                  "1.역사관광지 \t2.휴양관광지\t3.체험관광지\t4.문화시설\t5.건축/조형물\t6.자연관광지\t7.쇼핑"
                  "(ex) 1 2 3)")
    genre = genre.split()
    genre_0 = dest_dict[genre[0]]
    genre_1 = dest_dict[genre[1]]
    genre_2 = dest_dict[genre[2]]

    consider = input("고려할 사항을 선택하세요 (ex 1: 혼잡도, 2: 거리, 3: 혼잡도와 거리):")
    condition = input("추천 방법을 선택하세요 (ex 1:일 별 2: 개인 별 3: 개인 별 일별 4: 통합")

    print('-----------------------------추천 관광지 산출 중...-----------------------------')

    total_user_info = torch.LongTensor(total_user_info)
    total_user_info = total_user_info.to(device)
    print(total_user_info)

    # user_info = [u_id, age, sex, month, day, dow]
    u_id, age, sex = total_user_info.T[0], total_user_info.T[1], total_user_info.T[2]
    month, day, dow = total_user_info.T[3], total_user_info.T[4], total_user_info.T[5]

    u_embeds, _, _ = model(year=torch.LongTensor([0]),
                           u_id=u_id,
                           dow=dow,
                           pos_item=torch.LongTensor([0]),
                           neg_item=torch.empty(0),
                           node_flag=False)

    all_u_emb, all_i_emb = model.all_users_emb, model.all_items_emb
    all_pred_ratings = torch.mm(u_embeds, all_i_emb.T)
    all_rating, all_rank = torch.topk(all_pred_ratings, 100)

    recommend_des = []
    df_total = df_id_name_genre_coordinate[['destination', 'destination_name', 'genre', 'x', 'y']].copy()
    np.warnings.filterwarnings('ignore')

    dep_co = df_departure_coordinate.loc[df_departure_coordinate['dong'] == depart, ['x', 'y']]
    dep_co = (dep_co['x'], dep_co['y'])
    for item in df_total['destination'].unique():
        arr_co = df_total.loc[df_total['destination'] == item, ['x', 'y']]
        arr_co = (arr_co['x'], arr_co['y'])
        df_total.loc[df_total['destination'] == item, 'distance'] = haversine(dep_co, arr_co) * 1000

    for i in range(int(duration) * int(num)):
        d = i % int(duration)
        n = i // int(duration)

        # user_info = [u_id, age, sex, month, day, dow]
        u_info = total_user_info[i]
        df_total = df_total.loc[all_rank[i].tolist()]

        if 'rating' not in df_total.columns:
            df_total.loc[:, 'visitor'] = 0
        if 'day_' + str(d) not in df_total.columns:
            df_total.loc[:, 'day_' + str(d)] = 0
        if 'user_'+str(n) not in df_total.columns:
            df_total.loc[:, 'user_' + str(n)] = 0
        if 'user_' + str(n) + '_day_' + str(d) not in df_total.columns:
            df_total.loc[:, 'user_' + str(n) + '_day_' + str(d)] = 0

        df_total.loc[:, 'visitor'] = df_total.loc[:, 'visitor'] + all_rating[i].detach().cpu().clone().numpy()
        df_total.loc[:, 'user_' + str(n) + '_day_' + str(d)] = all_rating[i].detach().cpu().clone().numpy()
        df_total.loc[:, 'day_' + str(d)] = df_total.loc[:, 'day_' + str(d)] + all_rating[i].detach().cpu().clone().numpy()
        df_total.loc[:, 'user_' + str(n)] = df_total.loc[:, 'user_' + str(n)] + all_rating[i].detach().cpu().clone().numpy()

        # for daily personalized recommendation
        if consider == '1':
            df_congestion['congestion_1'] = df_congestion['congestion_1'] + np.ceil(np.abs(df_congestion['congestion_1'].min())+1)
            dest_congestion = df_congestion.loc[(df_congestion['month'] == u_info.tolist()[3]) &
                                                (df_congestion['day'] == u_info.tolist()[4])]
            df_congestion = df_congestion.sort_values(by='destination').reset_index().drop('index', axis=1)
            df_congestion = df_congestion.loc[all_rank[i].tolist()]

            ndcg_con = 1 / df_congestion['congestion_1'] / np.log2((np.array(rank2rate) + 1))
            df_total.loc[:, 'user_' + str(n) + '_day_' + str(d) + '_con'] = ndcg_con
            print('3', df_total)

        if consider == '2':
            dep_co = df_departure_coordinate.loc[df_departure_coordinate['dong'] == depart, ['x', 'y']]
            dep_co = (dep_co['x'], dep_co['y'])
            for item in df_total['destination'].unique():
                arr_co = df_total.loc[df_total['destination'] == item, ['x', 'y']]
                arr_co = (arr_co['x'], arr_co['y'])
                df_total.loc[df_total['destination'] == item, 'distance'] = haversine(dep_co, arr_co) * 1000

            ndcg_dis = 1 / df_total['distance'] / np.log2((np.array(rank2rate) + 1))
            df_total.loc[:, 'user_' + str(n) + '_day_' + str(d) + '_dis'] = ndcg_dis

        if consider == '3':
            df_congestion['congestion_1'] = df_congestion['congestion_1'] + np.ceil(np.abs(df_congestion['congestion_1'].min()) + 1)
            dest_congestion = df_congestion.loc[(df_congestion['month'] == u_info.tolist()[3]) &
                                                (df_congestion['day'] == u_info.tolist()[4])]
            df_congestion = df_congestion.sort_values(by='destination').reset_index().drop('index', axis=1)
            df_congestion = df_congestion.loc[all_rank[i].tolist()]

            ndcg_con = 1 / df_congestion['congestion_1'] / np.log2((np.array(rank2rate) + 1))
            df_total.loc[:, 'user_' + str(n) + '_day_' + str(d) + '_con'] = ndcg_con

            rank = df_total.sort_value(by='con').reset_index().index
            ndcg_dis = 1 / df_total['distance'] / np.log2((np.array(rank) + 1))
            df_total.loc[:, 'user_' + str(n) + '_day_' + str(d) + '_dis'] = ndcg_dis

    # 개인 별 일 별 데이터 생성 해놓음
    # 이 데이터로 날짜끼리 묶고 사람끼리 묶고 이 데이터 다 합쳐서 전체 출력
    # 일 별 개인별 통합 해야함ㅣ


    max = max(int(duration), int(num))
    for i in range(max):
        d = i % int(duration)
        n = i // int(duration)

    for i in range(int(duration)):









        # df_day.loc[:, 'day'+str(d)] = df_total.loc[:, 'day'+str(d)] + np.array(rank2rate)
        # df_user.loc[:, 'user'+str(n)] = df_total.loc[:, 'user'+str(n)] + np.array(rank2rate)

    df_genre = df_total.loc[(df_total['genre'] == genre_0) &
                            (df_total['genre'] == genre_1) &
                            (df_total['genre'] == genre_2)]

    df_genre = df_genre.sort_values(by='visitor').reset_index().drop('index', axis=1)

    for d in range(int(duration)):
        print(f'--------------여행 {d+1}일째 추천 여행지입니다.--------------')
        df_tmp = df_genre.loc[['destination_name', 'visitor']]
        df_tmp = df_genre.iloc[:int(rec_num)]
        print(df_tmp.reset_index().drop('index', axis=1))

"""

    if consider == '1':
        dest_congestion = df_congestion.loc[(df_congestion['month'] == u_info[3]) &
                                            (df_congestion['day'] == u_info[4])]
        df_congestion = df_congestion.sort_values(by='destination').reset_index().drop('index', axis=1)
        df_congestion = df_congestion[all_rank[i].tolist()]

        ndcg_con = df_congestion['congestion_1'] / np.log2((np.array(rank2rate) + 1))
        df_total.loc[:, 'con'] = df_total.loc[:,'user_' + str(n) + '_day_' + str(d)] + ndcg_con

    if consider == '2':
        dep_co = df_departure_coordinate.loc[df_departure_coordinate['dong'] == depart, ['x', 'y']]
        dep_co = (dep_co['x'], dep_co['y'])
        for item in df_total['destination'].unqiue():
            arr_co = df_total.loc[df_total['destination'] == item, ['x', 'y']]
            arr_co = (arr_co['x'], arr_co['y'])
            df_total.loc[df_total['destination'] == item, 'distance'] = haversine(dep_co, arr_co) * 1000
        df_total = df_total[]
        ndcg_dis = df_

    if consider == '3':
        pass

    # df_day.loc[:, 'day'+str(d)] = df_total.loc[:, 'day'+str(d)] + np.array(rank2rate)
    # df_user.loc[:, 'user'+str(n)] = df_total.loc[:, 'user'+str(n)] + np.array(rank2rate)

    df_genre = df_total.loc[(df_total['genre'] == dest_dict[genre[0]]) &
                            (df_total['genre'] == dest_dict[genre[1]]) &
                            (df_total['genre'] == dest_dict[genre[2]])]
"""