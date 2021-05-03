import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

def leakyrelu(x):
    x = torch.nn.functional.leaky_relu(x, negative_slope=1.0, inplace=False) 
    return x

class Aggregator(nn.Module):
    def __init__(self, sample_uu, sample_ui, sample_iu, sample_ii, sample_r, dim):
        super(Aggregator, self).__init__()
        self.Wusr = torch.nn.Linear(3 * dim, dim, bias=True).double()
        self.Witem = torch.nn.Linear(3 * dim, dim, bias=True).double()
        self.Wrelat = torch.nn.Linear(3 * dim, dim, bias=True).double()
        self.UTitem = torch.nn.Linear(1, dim, bias=True).double()
        self.UTuser = torch.nn.Linear(1, dim, bias=True).double()
        self.USpatial = torch.nn.Linear(3, dim, bias=True).double()
        self.sample_uu = sample_uu
        self.sample_ui = sample_ui
        self.sample_iu = sample_iu
        self.sample_ii = sample_ii
        self.sample_r = sample_r
        self.bn1 = torch.nn.BatchNorm1d(dim).double()
        self.bn2 = torch.nn.BatchNorm1d(dim).double()
        self.bn3 = torch.nn.BatchNorm1d(dim).double()
        self.bn4 = torch.nn.BatchNorm1d(dim).double()
        self.bn5 = torch.nn.BatchNorm1d(dim).double()
        self.bn6 = torch.nn.BatchNorm1d(dim).double()
        # self.UTitem.weight.data.fill_(-0.000001)
        # self.UTuser.weight.data.fill_(-0.000001)
        # self.UTitem.bias.data.fill_(1)
        # self.UTuser.bias.data.fill_(1)

        # self.user_position = user_position
        # self.social_graph = social_graph
        # self.usertoitem_graph = usertoitem_graph
        # self.itemtouser_graph = itemtouser_graph
        # self.item_kg = item_kg
        # self.relationships = relationships
        # self.n_user = n_user
        # self.n_item = n_item
        # self.n_relation = n_relation
        # self.isSocial = isSocial
        # self.isKg = isKg
        # self.isTime = isTime
        # self.isPos = isPos
        self.dim = dim
        # self.max_rating = max_rating

        

    # def forward( self, usr_feat, item_feat, rel_feat):
    #     pass
    def forward(self, act, user_emb, item_emb, relat_emb, uu_emb, uu_st, uu_pos, ui_emb, ui_rat, ui_vot, ui_tim, iu_emb, iu_rat, iu_vot, iu_tim, ii_emb, ii_rel, rl_emb, rr_emb):
        # print(user_emb.size(),  item_emb.size(), relat_emb.size(), uu_emb.size(), uu_st.size(), uu_pos.size(), ui_emb.size(), ui_rat.size(), ui_vot.size(), ui_tim.size(), iu_emb.size(), iu_rat.size(), iu_vot.size(), iu_tim.size(), ii_emb.size(), ii_rel.size(), rl_emb.size(), rr_emb.size())
        # torch.Size([1, 4]) torch.Size([48, 4]) torch.Size([25, 4]) torch.Size([3, 4]) torch.Size([3]) torch.Size([3, 3]) torch.Size([4, 4]) torch.Size([4]) torch.Size([4]) torch.Size([4]) torch.Size([144, 4]) torch.Size([144]) torch.Size([144]) torch.Size([144]) torch.Size([144, 4]) torch.Size([144, 4]) torch.Size([100, 4]) torch.Size([100, 4])
        # torch.Size([1, 4]) torch.Size([12, 4]) torch.Size([0, 4]) torch.Size([3, 4]) torch.Size([3]) torch.Size([3, 3]) torch.Size([4, 4]) torch.Size([4]) torch.Size([4]) torch.Size([4]) torch.Size([36, 4]) torch.Size([36]) torch.Size([36]) torch.Size([36]) torch.Size([36, 4]) torch.Size([36, 4]) torch.Size([0, 4]) torch.Size([0, 4])

        #useruser
        temp = torch.repeat_interleave(uu_st.view((-1, self.sample_uu)).sum(dim = 1), self.sample_uu, dim = 0)
        # print(uu_st)
        # print(temp)
        temp[temp==0]=1.0
        # print(temp)
        user_neigh_ur = uu_emb*torch.sigmoid((self.bn4(self.USpatial(uu_pos.double()))))*((uu_st/temp).unsqueeze(dim=1))
        user_neigh_ur = user_neigh_ur.view((-1, self.sample_uu, self.dim)).sum(dim=1)
        # print(user_neigh_ur)

        #useritem
        temp = ui_rat*ui_vot
        temp1 = torch.repeat_interleave(temp.view((-1, self.sample_ui)).sum(dim = 1), self.sample_ui, dim = 0)
        temp1[temp1==0]=1.0
        user_neigh_it = ui_emb*torch.sigmoid((self.bn5(self.UTuser(ui_tim.view(ui_emb.size()[0], 1).double()))))*((temp/temp1).unsqueeze(dim=1))
        user_neigh_it = user_neigh_it.view((-1, self.sample_ui, self.dim)).sum(dim=1)

        #user
        user_agg = torch.cat((user_emb, user_neigh_it, user_neigh_ur), 1)
        user_agg = self.bn1(self.Wusr(user_agg))
        # print("helo1")
        # print(user_agg)
        user_agg = act(user_agg)
        # print(user_agg)

        #item user
        temp = iu_rat*iu_vot
        temp1 = torch.repeat_interleave(temp.view((-1, self.sample_iu)).sum(dim = 1), self.sample_iu, dim = 0)
        temp1[temp1==0]=1.0
        item_neigh_ur = iu_emb*torch.sigmoid((self.bn6(self.UTitem(iu_tim.view(iu_emb.size()[0], 1).double()))))*((temp/temp1).unsqueeze(dim=1))
        item_neigh_ur = item_neigh_ur.view((-1, self.sample_iu, self.dim)).sum(dim=1)
        
        #itemitem
        temp = torch.exp((torch.repeat_interleave(item_emb, self.sample_ii, dim = 0)* ii_rel).sum(dim = 1))
        temp1 = torch.repeat_interleave(temp.view((-1, self.sample_ii)).sum(dim = 1), self.sample_ii, dim = 0)
        temp1[temp1==0]=1.0
        item_neigh_it = ii_emb*((temp/temp1).unsqueeze(dim=1))
        item_neigh_it = item_neigh_it.view((-1, self.sample_ii, self.dim)).sum(dim=1)
        
        #item
        item_agg = torch.cat((item_emb, item_neigh_it, item_neigh_ur), 1)
        item_agg = self.bn2(self.Witem(item_agg))
        # print("helo2")
        # print(item_agg)
        item_agg = act(item_agg)
        # print(item_agg)

        #relation_left
        relat_st = torch.mean(rl_emb.view((-1, self.sample_r, self.dim)), dim=1)
        relat_en = torch.mean(rr_emb.view((-1, self.sample_r, self.dim)), dim=1)

        #relation
        relat_agg = torch.cat((relat_emb, relat_st, relat_en), 1)
        relat_agg = self.bn3(self.Wrelat(relat_agg.double()))
        relat_agg = act(relat_agg)
        # print(user_agg, item_agg, relat_agg)
        
        return user_agg, item_agg, relat_agg