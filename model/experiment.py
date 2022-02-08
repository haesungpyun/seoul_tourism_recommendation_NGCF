from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

class Experiment():
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim,
                 criterion: nn.Module,
                 train_dataloader: torch.utils.data.DataLoader,
                 test_dataloader: torch.utils.data.DataLoader,
                 epochs: int,
                 ks: int,
                 device):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.epochs = epochs
        self.ks = ks
        self.device = device
        self.rmse = RMSELoss()

    def train(self):
        print('------------------------- Train -------------------------')
        for epoch in range(self.epochs):
            total_loss = 0
            d1 = datetime.now()
            for year, u_id, age, date, sex, pos_item, neg_item in self.train_dataloader:
                year, u_id = year.to(self.device), u_id.to(self.device)
                age, date, sex = age.to(self.device), date.to(self.device), sex.to(self.device)
                pos_item, neg_item = pos_item.to(self.device), neg_item.to(self.device)

                u_embeds, pos_i_embeds, neg_i_embeds = self.model(year=year,
                                                                  u_id=u_id,
                                                                  age=age,
                                                                  date=date,
                                                                  sex=sex,
                                                                  pos_item=pos_item,
                                                                  neg_item=neg_item,
                                                                  node_flag=True)
                self.optimizer.zero_grad()
                loss = self.criterion(u_embeds, pos_i_embeds, neg_i_embeds)
                loss.backward()
                self.optimizer.step()
                total_loss += loss
            HR, NDCG, RMSE = self.eval()
            print(f'epoch {epoch + 1}, epoch loss: {total_loss / len(self.train_dataloader)}, HR:{HR}, NDCG:{NDCG}, RMSE:{RMSE}, Run time:{datetime.now()-d1}')

    def eval(self):
        NDCG = []
        HR = []
        RMSE = 0
        with torch.no_grad():
            self.model.eval()
            for year, u_id, age, date, sex, congestion, pos_item in self.test_dataloader:
                year, u_id, pos_item = year.to(self.device), u_id.to(self.device), pos_item.to(self.device)
                age, date, sex = age.to(self.device), date.to(self.device), sex.to(self.device)
                congestion = congestion.to(self.device)

                u_embeds, pos_i_embeds, _ = self.model(year=year,
                                                       u_id=u_id,
                                                       age=age,
                                                       date=date,
                                                       sex=sex,
                                                       pos_item=pos_item,
                                                       neg_item=torch.empty(0),
                                                       node_flag=False)

                gt_rank = pos_item[0].item()
                pred_ratings = torch.mm(u_embeds, pos_i_embeds.T)

                # HR
                _, pred_rank = torch.topk(pred_ratings[0], 5)
                recommends_HR = torch.take(pos_item, pred_rank).cpu().numpy().tolist()
                HR.append(self.hit(gt_item=gt_rank, pred_items=recommends_HR))
                # NDCG
                _, pred_rank = torch.topk(pred_ratings[0], self.ks)
                recommends_NDCG = torch.take(pos_item, pred_rank).cpu().numpy().tolist()
                NDCG.append(self.Ndcg(gt_item=gt_rank, pred_items=recommends_NDCG))
                # RMSE
                pred_rate = pred_ratings[0,0]
                pred_rate = pred_rate.to(self.device)
                RMSE += self.rmse(pred_rate, congestion)
        return np.mean(HR), np.mean(NDCG), RMSE / len(self.test_dataloader)


    def Ndcg(self, gt_item, pred_items):
        # IDCG = self.dcg(gt_items)
        # DCG = self.dcg(pred_items)
        # output = DCG/IDCG
        if gt_item in pred_items:
            index = pred_items.index(gt_item)
            return np.reciprocal(np.log2(index + 2))
        return 0

    def hit(self, gt_item, pred_items):
        if gt_item in pred_items:
            return 1
        return 0

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,pred,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(pred, y))
        return loss
