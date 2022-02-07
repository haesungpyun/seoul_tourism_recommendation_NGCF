from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np



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

    def train(self):
        print('------------------------- Train -------------------------')

        for epoch in range(self.epochs):
            total_loss = 0
            for year, u_id, age, day, sex, pos_item, neg_item in self.train_dataloader:
                year, u_id = year.to(self.device), u_id.to(self.device)
                age, day, sex = age.to(self.device), day.to(self.device), sex.to(self.device)
                pos_item, neg_item = pos_item.to(self.device), neg_item.to(self.device)

                u_embeds, pos_i_embeds, neg_i_embeds = self.model(year=year,
                                                                  u_id=u_id,
                                                                  age=age,
                                                                  day=day,
                                                                  sex=sex,
                                                                  pos_item=pos_item,
                                                                  neg_item=neg_item,
                                                                  node_flag=True)
                self.optimizer.zero_grad()
                loss = self.criterion(u_embeds, pos_i_embeds, neg_i_embeds)
                loss.backward()
                self.optimizer.step()
                total_loss += loss

            HR, NDCG = self.eval()
            print(f'epoch {epoch + 1}, epoch loss: {total_loss / len(self.train_dataloader)}, HR:{HR}, NDCG:{NDCG}')

    def eval(self):
        NDCG = []
        HR = []
        with torch.no_grad():
            for year, u_id, age, day, sex, pos_item in self.test_dataloader:
                year, u_id, pos_item = year.to(self.device), u_id.to(self.device), pos_item.to(self.device)
                age, day, sex = age.to(self.device), day.to(self.device), sex.to(self.device)

                u_embeds, pos_i_embeds, _ = self.model(year=year,
                                                       u_id=u_id,
                                                       age=age,
                                                       day=day,
                                                       sex=sex,
                                                       pos_item=pos_item,
                                                       neg_item=torch.empty(0),
                                                       node_flag=False)

                # all_i_emb = self.model.all_items_emb
                # all_pred_ratings = torch.mm(u_embeds, all_i_emb.T)
                # _, all_rank = torch.topk(all_pred_ratings[0], self.ks)
                gt_rank = pos_item[0].item()
                pred_ratings = torch.mm(u_embeds, pos_i_embeds.T)

                # HR
                _, pred_rank = torch.topk(pred_ratings[0], 3)
                recommends_HR = torch.take(pos_item, pred_rank).cpu().numpy().tolist()
                HR.append(self.hit(gt_item=gt_rank, pred_items=recommends_HR))


                # NDCG
                _, pred_rank = torch.topk(pred_ratings[0], self.ks)
                recommends_NDCG = torch.take(pos_item, pred_rank).cpu().numpy().tolist()
                NDCG.append(self.Ndcg(gt_item=gt_rank, pred_items=recommends_NDCG))

        return np.mean(HR), np.mean(NDCG)

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