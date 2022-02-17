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
import time


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


def map_func(b):
    return item_dict[b]


vec_func = np.vectorize(map_func)

if __name__ == '__main__':
    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # print GPU information
    if torch.cuda.is_available():
        print('Current cuda device:', torch.cuda.current_device())
        print('Count of using GPUs:', torch.cuda.device_count())

    FOLDER_PATH = 'saved_model_data'

    d1 = time.time()
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
    df_id_name_genre_coordinate.loc[:, 'itemid'] = vec_func(df_id_name_genre_coordinate['destination'])
    df_id_name_genre_coordinate = df_id_name_genre_coordinate.rename(columns={'middle_category_name': 'genre'})

    PATH = os.path.join(root_dir, 'congestion_1_2' + '.pkl')
    with open(PATH, 'rb') as f:
        df_congestion = CPU_Unpickler(f).load()
    df_congestion = pd.pivot_table(df_congestion, index=['month', 'day', 'dayofweek', 'destination'],
                                   aggfunc={'congestion_1': 'sum',
                                            'congestion_2': 'sum'})
    df_congestion = df_congestion.reset_index()
    df_congestion.loc[:, 'itemid'] = vec_func(df_congestion['destination'])

    PATH = os.path.join(root_dir, 'seoul_gu_dong_coordinate' + '.pkl')
    with open(PATH, 'rb') as f:
        df_departure_coordinate = CPU_Unpickler(f).load()
    print("Destination Data Loaded!")

    num_list = ['첫', '두', '세', '네']
    week = ['월', '화', '수', '목', '금', '토', '일']
    gender = ['여', '남']
    month_info = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    dest_dict = {'1': '역사관광지', '2': '휴양관광지', '3': '체험관광지', '4': '문화시설', '5': '건축/조형물', '6': '자연관광지', '7': '쇼핑'}
    rank2rate = []
    for i in range(100):
        rank2rate.append(i)

    print("Data Load time: ",  time.time()-d1)

    print('---------------------------------------------------------------------------------------------')
    print("관광객 수를 입력하세요(ex 2):")
    num = input()
    print("관광 기간를 입력하세요(ex 7):")
    duration = input()
    print("관광할 시작 월-일-요일을 입력하세요(ex 01 01 수):")
    dates = input().split()
    print("추천 받을 관광지의 개수를 입력하세요(ex 10)")
    rec_num = input()
    print('출발지를 입력하세요 (ex) 사직동)')
    depart = input()

    if (dates[0] =='02') & (dates[1] =='29'):
        dates[1] = '28'

    month = torch.LongTensor([int(dates[0])]).to(device)
    day = torch.LongTensor([int(dates[1])]).to(device)
    dow = torch.LongTensor([week.index(dates[2])]).to(device)

    total_user_info = []
    for i in range(int(num)):
        print(f'{num_list[i]}번째 관광객의 성별을 입력하세요(ex 남/여):')
        sex = input()
        print(f'{num_list[i]}번째 관광객의 연령을 입력하세요(ex 23):')
        age = input()
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

    print("관광지의 유형을 선택하세요\n"
          "1.역사관광지 \t2.휴양관광지\t3.체험관광지\t4.문화시설\t5.건축/조형물\t6.자연관광지\t7.쇼핑 (ex) 1 2 3):")
    genre = input().split()
    genre = genre.split()
    genre_0 = dest_dict[genre[0]]
    genre_1 = dest_dict[genre[1]]
    genre_2 = dest_dict[genre[2]]

    print("선호도 혼잡도 거리의 중요도 비율을 입력하세요 (ex) 선호도 0.5 혼잡도 0.3 거리 0.2의 비율로 고려 -> 0.5 0.2 0.3)")
    condition = input().split()
    condition = condition.split()
    vis_rat = float(condition[0])
    con_rat = float(condition[1])
    dis_rat = float(condition[2])

    print("추천 방법을 선택하세요 (ex 1:날짜 별 2: 성별 나이 별 추천 3: 날짜 성별 나이 별 4: 종합")
    rec_type = input()


    d1 = time.time()
    print('-----------------------------추천 관광지 산출 중...-----------------------------')
    total_user_info = torch.LongTensor(total_user_info)

    # total_user_info = torch.unique(total_user_info, dim=0)
    total_user_info = total_user_info.to(device)
    print(total_user_info)

    # user_info = [u_id, age, sex, month, day, dow]
    u_id, age, sex = total_user_info.T[0], total_user_info.T[1], total_user_info.T[2],
    month, day, dow = total_user_info.T[3], total_user_info.T[4], total_user_info.T[5],

    u_embeds, _, _ = model(year=torch.LongTensor([0]),
                           u_id=u_id,
                           dow=dow,
                           pos_item=torch.LongTensor([0]),
                           neg_item=torch.empty(0),
                           node_flag=False)

    all_u_emb, all_i_emb = model.all_users_emb, model.all_items_emb
    all_pred_ratings = torch.mm(u_embeds, all_i_emb.T)
    all_rating, all_rank = torch.topk(all_pred_ratings, 100)

    df_total = df_id_name_genre_coordinate[['destination_name', 'genre', 'x', 'y', 'itemid']].copy()
    df_total = df_total.set_index('itemid')

    dep_co = df_departure_coordinate.loc[df_departure_coordinate['dong'] == depart, ['x', 'y']]
    dep_co = (dep_co['x'], dep_co['y'])
    for item in df_total['destination_name'].unique():
        arr_co = df_total.loc[df_total['destination_name'] == item, ['x', 'y']]
        arr_co = (arr_co['x'], arr_co['y'])
        df_total.loc[df_total['destination_name'] == item, 'distance'] = haversine(dep_co, arr_co) * 1000
    df_total = df_total.drop(columns=['x', 'y'])

    df_total = df_total.reset_index()[['itemid', 'destination_name', 'distance', 'genre']]
    df_day = df_total.reset_index()[['itemid', 'destination_name', 'distance', 'genre']]
    df_day = df_day.set_index('itemid')
    df_user = df_total.reset_index()[['itemid', 'destination_name', 'distance', 'genre']]
    df_user = df_user.set_index('itemid')
    df_daily_user = df_total.reset_index()[['itemid', 'destination_name', 'distance', 'genre']]
    df_daily_user = df_daily_user.set_index('itemid')
    df_total = df_total.set_index('itemid')

    for i in range(len(total_user_info)):
        # user_info = [u_id, age, sex, month, day, dow]
        u_info = total_user_info[i]
        u_id = u_info[0].item()
        age = u_info[1].item()
        sex = u_info[2].item()
        month = u_info[3].item()
        day = u_info[4].item()
        dow = u_info[5].item()

        df_con_tmp = df_congestion.loc[(df_congestion['month'] == u_info.tolist()[3]) &
                                       (df_congestion['day'] == u_info.tolist()[4]) &
                                       (df_congestion['dayofweek'] == u_info.tolist()[5])]
        df_con_tmp = df_con_tmp.set_index('itemid')
        df_con_tmp = df_con_tmp.sort_values(by='congestion_1')

        if 'rating' not in df_total.columns:
            df_total.loc[:, 'visitor'] = 0
        if str(u_id) not in df_user.columns:
            df_daily_user.loc[:, str(u_id)] = 0
        if str(month) + '-' + str(day) not in df_day.columns:
            df_day.loc[:, str(month) + '-' + str(day)] = 0
        if str(age)+'-'+str(sex) not in df_user.columns:
            df_user.loc[:, str(age)+'-'+str(sex)] = 0

        df_total = df_total.loc[all_rank[i].tolist()]
        df_total.loc[:, 'visitor'] = df_total.loc[:, 'visitor'] + (np.array(rank2rate) * vis_rat)
        df_total = df_total.loc[df_con_tmp.index]
        df_total.loc[:, 'visitor'] = df_total.loc[:, 'visitor'] + (np.array(rank2rate) * con_rat)
        df_total = df_total.sort_values(by='distance')
        df_total.loc[:, 'visitor'] = df_total.loc[:, 'visitor'] + (np.array(rank2rate) * dis_rat)

        df_day = df_day.loc[all_rank[i].tolist()]
        df_day.loc[:, str(month) + '-' + str(day)] = df_day.loc[:, str(month) + '-' + str(day)] + \
                                                     (np.array(rank2rate) * vis_rat)
        df_day = df_day.loc[df_con_tmp.index]
        df_day.loc[:, str(month) + '-' + str(day)] = df_day.loc[:, str(month) + '-' + str(day)] + \
                                                     (np.array(rank2rate) * con_rat)
        df_day = df_day.sort_values(by='distance')
        df_day.loc[:, str(month) + '-' + str(day)] = df_day.loc[:, str(month) + '-' + str(day)] + \
                                                     (np.array(rank2rate) * dis_rat)

        df_user = df_user.loc[all_rank[i].tolist()]
        df_user.loc[:, str(age)+'-'+str(sex)] = df_user.loc[:, str(age)+'-'+str(sex)] + (np.array(rank2rate) * vis_rat)
        df_user = df_user.loc[df_con_tmp.index]
        df_user.loc[:, str(age)+'-'+str(sex)] = df_user.loc[:, str(age)+'-'+str(sex)] + (np.array(rank2rate) * con_rat)
        df_user = df_user.sort_values(by='distance')
        df_user.loc[:, str(age)+'-'+str(sex)] = df_user.loc[:, str(age)+'-'+str(sex)] + (np.array(rank2rate) * dis_rat)

        df_daily_user = df_daily_user.loc[all_rank[i].tolist()]
        df_daily_user.loc[:, str(u_id)] = df_daily_user.loc[:, str(u_id)] + (np.array(rank2rate) * vis_rat)
        df_daily_user = df_daily_user.loc[df_con_tmp.index]
        df_daily_user.loc[:, str(u_id)] = df_daily_user.loc[:, str(u_id)] + (np.array(rank2rate) * con_rat)
        df_daily_user = df_daily_user.sort_values(by='distance')
        df_daily_user.loc[:, str(u_id)] = df_daily_user.loc[:, str(u_id)] + (np.array(rank2rate) * dis_rat)

    df_day = df_day.loc[(df_total['genre'] == genre_0) |
                        (df_total['genre'] == genre_1) |
                        (df_total['genre'] == genre_2)]

    df_user = df_user.loc[(df_total['genre'] == genre_0) |
                        (df_total['genre'] == genre_1) |
                        (df_total['genre'] == genre_2)]

    df_daily_user = df_daily_user.loc[(df_total['genre'] == genre_0) |
                          (df_total['genre'] == genre_1) |
                          (df_total['genre'] == genre_2)]

    while rec_type != '5':

        if rec_type == '1':
            for col in df_day.iloc[:, 3:].columns:
                df_ge_med_bool = df_day[col].ge(np.floor(df_day.iloc[:, 3:].median(axis=1)), axis=0)
                idx = df_day[col][df_ge_med_bool].index

                print(f"-------------------{col.split('-')[0]}월 {col.split('-')[1]}일 추천 여행지입니다.-------------------")
                tmp = df_day.loc[idx][['destination_name', col]]
                print(tmp.sort_values(by=col, ascending=False).iloc[:10].reset_index().drop('itemid', axis=1))

            print("Recommend Run time: ", time.time()-d1)
            print("추천 방법을 선택하세요 (ex 1:날짜 별 2: 성별 나이 별 추천 3: 날짜 성별 나이 별 4: 종합 5: 종료 ")
            rec_type = input()
            d1 = time.time()

        if rec_type == '2':
            for col in df_user.iloc[:, 3:].columns:
                df_ge_med_bool = df_user[col].ge(np.floor(df_user.iloc[:, 3:].median(axis=1)), axis=0)
                idx = df_user[col][df_ge_med_bool].index
                if col.split('-')[1] == '0':
                    sex = '여성'
                else:
                    sex = '남성'
                print(f"-------------------{int(col.split('-')[0])-5}대 {sex}분 추천 여행지입니다.-------------------")
                tmp = df_user.loc[idx][['destination_name', col]]
                print(tmp.sort_values(by=col, ascending=False).iloc[:10].reset_index().drop('itemid', axis=1))

            print("Recommend Run time: ", time.time() - d1)
            print("추천 방법을 선택하세요 (ex 1:날짜 별 2: 성별 나이 별 추천 3: 날짜 성별 나이 별 4: 종합 5: 종료 ")
            rec_type = input()
            d1 = time.time()

        if rec_type == '3':
            for col in df_daily_user.iloc[:, 3:].columns:
                df_ge_med_bool = df_daily_user[col].ge(np.floor(df_daily_user.iloc[:, 3:].median(axis=1)), axis=0)
                idx = df_daily_user[col][df_ge_med_bool].index

                info = list(user_dict.keys())[list(user_dict.values()).index(int(col))]
                age = info[:2]
                sex = info[2]
                month = info[3:5]
                day = info[5:7]
                if sex == '0':
                    s = '여성'
                else:
                    s = '남성'
                print(f"-------------------{int(age)-5}대 {s}분의 {month}월 {day}일 추천 여행지입니다.-------------------")
                tmp = df_daily_user.loc[idx][['destination_name', col]]
                print(tmp.sort_values(by=col, ascending=False).iloc[:10].reset_index().drop('itemid', axis=1))

            print("Recommend Run time: ", time.time() - d1)
            print("추천 방법을 선택하세요 (ex 1:날짜 별 2: 성별 나이 별 추천 3: 날짜 성별 나이 별 4: 종합 5: 종료 ")
            rec_type = input()
            d1 = time.time()

        if rec_type == '4':
            for col in df_total.iloc[:, 3:].columns:
                df_ge_med_bool = df_total[col].ge(np.floor(df_total.iloc[:, 3:].median(axis=1)), axis=0)
                idx = df_total[col][df_ge_med_bool].index

                print(f"-------------------여행 기간 {int(duration)}일 간 추천 여행지입니다.-------------------")
                tmp = df_total.loc[idx][['destination_name', col]]
                print(tmp.sort_values(by=col, ascending=False).iloc[:10].reset_index().drop('itemid', axis=1))

            print("Recommend Run time: ", time.time() - d1)
            print("추천 방법을 선택하세요 (ex 1:날짜 별 2: 성별 나이 별 추천 3: 날짜 성별 나이 별 4: 종합 5: 종료 ")
            rec_type = input()
            d1 = time.time()
