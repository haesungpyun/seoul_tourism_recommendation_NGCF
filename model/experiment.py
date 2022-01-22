from torch.utils.data import Dataset
import torch
import torch.nn as nn
import os
import numpy as np
from parsers import args


class Train():
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim,
                 criterion: nn.Module,
                 train_dataloader: torch.utils.data.DataLoader,
                 test_dataloader: torch.utils.data.DataLoader,
                 epochs: int,
                 device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.epochs = epochs
        self.device = device

    def train(self):
        for epoch in range(self.epochs):

            total_loss = 0

            for date, u_id, u_feats, pos_item, neg_item in self.train_dataloader:
                date, u_id, u_feats = date.to(self.device), u_id.to(self.device), u_feats.to(self.device)
                pos_item, neg_item = pos_item.to(self.device), neg_item.to(self.device)

                print('Train')
                print('date', date.shape, date)
                print('u_id', u_id.shape, u_id)
                print('u_feats', u_feats.shape, u_feats)
                print('pos_item', pos_item.shape, pos_item)
                print('neg_item', neg_item.shape, neg_item)

                u_embeds, pos_i_embeds, neg_i_embeds = self.model(dateidx=date,
                                                                  user_idx=u_id,
                                                                  u_feats=u_feats,
                                                                  pos_item=pos_item,
                                                                  neg_item=neg_item,
                                                                  node_flag=True)
                self.optimizer.zero_grad()
                loss = self.criterion(u_embeds, pos_i_embeds, neg_i_embeds)
                loss.backward()
                self.optimizer.step()
                total_loss += loss
                break

            print('epoch {}'.format(epoch + 1))

            test = Test(model=self.model,
                        dataloader=self.test_dataloader,
                        ks=args.ks,
                        device=self.device)

            print('|epoch loss: {}|'.format((total_loss / len(self.train_dataloader))))
            test.eval()


class Test():
    def __init__(self,
                 model: nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 ks: int,
                 device):
        self.model = model
        self.dataloader = dataloader
        self.ks = ks
        self.device = device

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

    def eval(self):
        NDCG = []
        HR = []
        with torch.no_grad():
            for date, u_id, u_feats, pos_item in self.dataloader:
                date, u_id, u_feats, pos_item = date.to(self.device), u_id.to(self.device), u_feats.to(
                    self.device), pos_item.to(self.device)

                print('Test')
                print('date', date.shape, date)
                print('u_id', u_id.shape, u_id)
                print('u_feats', u_feats.shape, u_feats)
                print('pos_item', pos_item.shape, pos_item)

                u_embeds, pos_i_embeds, _ = self.model(dateidx=date,
                                                       user_idx=u_id,
                                                       u_feats=u_feats,
                                                       pos_items=pos_item,
                                                       neg_items=torch.empty(0),
                                                       node_flag=False)

                pred_ratings = torch.mm(u_embeds, pos_i_embeds.T)
                _, pred_rank = torch.topk(pred_ratings[0], self.ks)

                print('pred_rank', pred_rank)

                recommends = torch.take(
                    pos_item, pred_rank).cpu().numpy().tolist()

                print('recommend', recommends)

                gt_rank = pos_item[0].item()
                print('gt_rank', gt_rank)

                HR.append(self.hit(gt_item=gt_rank, pred_items=recommends))
                NDCG.append(self.Ndcg(gt_item=gt_rank, pred_items=recommends))
                print('HR:{}, NDCG:{}'.format((HR), (NDCG)))
                break

        print('HR:{}, NDCG:{}'.format(np.mean(HR), np.mean(NDCG)))


