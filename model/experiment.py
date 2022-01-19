from torch.utils.data import Dataset
import torch as t
import torch.nn as nn
import os
import numpy as np
from parsers import args


class Train():
    def __init__(self,
                 model: nn.Module,
                 optimizer: t.optim,
                 criterion: nn.Module,
                 train_dataloader: t.utils.data.DataLoader,
                 test_dataloader: t.utils.data.DataLoader,
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

            for u_id, pos_item, neg_item in self.train_dataloader:
                u_id, pos_item, neg_item = u_id.to(self.device), pos_item.to(self.device), neg_item.to(self.device)
                u_embeds, pos_i_embeds, neg_i_embeds = self.model(u_id, pos_item, neg_item, True)

                self.optimizer.zero_grad()
                loss = self.criterion(u_embeds, pos_i_embeds, neg_i_embeds)
                loss.backward()
                self.optimizer.step()
                total_loss += loss

            print('epoch {}'.format(epoch + 1))

            test = Test(model=self.model,
                        dataloader=self.test_dataloader,
                        ks=args.ks,
                        device=self.device)

            print('|epoch loss: {}|'.format((total_loss/len(self.train_dataloader))))
            test.eval()


class Test():
    def __init__(self,
                 model: nn.Module,
                 dataloader: t.utils.data.DataLoader,
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
        with t.no_grad():
            for u_id, pos_items in self.dataloader:
                u_id, pos_items = u_id.to(self.device), pos_items.to(self.device)

                u_embeds, pos_i_embeds, _ = self.model(users=u_id,
                                                       pos_items=pos_items,
                                                       neg_items=t.empty(0),
                                                       node_flag=False)

                pred_ratings = t.mm(u_embeds, pos_i_embeds.T)
                _, pred_rank = t.topk(pred_ratings[0], self.ks)

                recommends = t.take(
                    pos_items, pred_rank).cpu().numpy().tolist()

                gt_rank = pos_items[0].item()

                HR.append(self.hit(gt_item=gt_rank, pred_items=recommends))
                NDCG.append(self.Ndcg(gt_item=gt_rank, pred_items=recommends))

        print('HR:{}, NDCG:{}'.format(np.mean(HR), np.mean(NDCG)))


