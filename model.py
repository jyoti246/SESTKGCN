import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from aggregator import Aggregator
from sklearn.preprocessing import LabelEncoder

class SESTKGCN(nn.Module):
    def __init__(self, df_position, df_sg, df_useritem, df_itemuser, df_kg, df_krelat, n_user, n_item, n_relation, dim, n_layers, max_rating,  isSocial, isKg, isTime, isPos, sample_uu, sample_ui, sample_iu, sample_ii, sample_r, device):
        super(SESTKGCN, self).__init__()
        self.df_position = df_position
        self.df_sg = df_sg
        self.df_useritem = df_useritem
        self.df_itemuser = df_itemuser
        self.df_kg = df_kg
        self.df_krelat = df_krelat
        # self.relationships = relationships
        self.n_user = n_user
        self.n_item = n_item
        self.n_relation = n_relation
        self.dim = dim
        self.n_layers = n_layers
        self.max_rating = max_rating
        self.sample_uu = sample_uu
        self.sample_ui = sample_ui
        self.sample_iu = sample_iu
        self.sample_ii = sample_ii
        self.sample_r = sample_r
        self.isPos = isPos
        self.isSocial = isSocial
        self.isTime = isTime
        self.isKg = isKg


        self.usr_feat = torch.nn.Embedding(n_user, dim).weight
        self.item_feat = torch.nn.Embedding(n_item, dim).weight
        self.rel_feat = torch.nn.Embedding(n_relation, dim).weight
        # self.usr_feat = torch.empty(n_user, dim)
        # self.item_feat = torch.empty(n_item, dim)
        # self.rel_feat = torch.empty(n_relation, dim)
        torch.nn.init.normal_(self.usr_feat, std = .8)
        torch.nn.init.normal_(self.item_feat, std = .8)
        torch.nn.init.normal_(self.rel_feat, std = .8)
        # print(self.usr_feat)
        # print(self.item_feat)
        # print(self.rel_feat)
        
        # self.usr_feat.requires_grad=True
        # self.item_feat.requires_grad=True
        # self.rel_feat.requires_grad=True

        self.layers = nn.ModuleList()
        self.layers.append(Aggregator(sample_uu, sample_ui, sample_iu, sample_ii, sample_r, dim))
        for i in range(1, n_layers ):
            self.layers.append(Aggregator(sample_uu, sample_ui, sample_iu, sample_ii, sample_r, dim))

        self.neigh = [] # 14 ka gap
        curr_u = list(range(n_user))
        curr_v = list(range(n_item))
        curr_r = list(range(n_relation))
        random.seed(3)

        for i in range(1):
            neigh_uu = []
            neigh_ui = []
            neigh_uu_st = []
            neigh_ui_rat = []
            neigh_ui_vot = []
            neigh_ui_tim = []
            for x in curr_u:
                # print(x)
                if len(self.df_sg[x])>= self.sample_uu:
                    neighbors = random.sample(self.df_sg[x], self.sample_uu)
                    neigh_uu.extend([nei for nei, _ in neighbors])
                    neigh_uu_st.extend([nei for _, nei in neighbors])
                elif len(self.df_sg[x]) == 0:
                    neigh_uu.extend([0]*self.sample_uu)
                    neigh_uu_st.extend([0.0]*self.sample_uu)
                else:
                    neighbors = random.choices(self.df_sg[x], k = self.sample_uu)
                    neigh_uu.extend([nei for nei, _ in neighbors])
                    neigh_uu_st.extend([nei for _, nei in neighbors])

                if len(self.df_useritem[x])>= self.sample_ui:
                    neighbors = random.sample(self.df_useritem[x], self.sample_ui)
                    for nei, rat, vot, tim in neighbors:
                        neigh_ui.append(nei)
                        neigh_ui_rat.append(rat)
                        neigh_ui_vot.append(vot)
                        neigh_ui_tim.append(tim)
                elif len(self.df_useritem[x]) == 0:
                    neigh_ui.extend([0]*self.sample_ui)
                    neigh_ui_rat.extend([0.0]*self.sample_ui)
                    neigh_ui_vot.extend([0.0]*self.sample_ui)
                    neigh_ui_tim.extend([0.0]*self.sample_ui)
                else:
                    neighbors = random.choices(self.df_useritem[x], k = self.sample_ui)
                    for nei, rat, vot, tim in neighbors:
                        neigh_ui.append(nei)
                        neigh_ui_rat.append(rat)
                        neigh_ui_vot.append(vot)
                        neigh_ui_tim.append(tim)

            neigh_iu = []
            neigh_ii = []
            neigh_ir = []
            neigh_iu_rat = []
            neigh_iu_vot = []
            neigh_iu_tim = []
            for x in curr_v:
                if len(self.df_kg[x])>= self.sample_ii:
                    neighbors = random.sample(self.df_kg[x], self.sample_ii)
                    neigh_ii.extend([nei for nei, _ in neighbors])
                    neigh_ir.extend([nei for  _, nei in neighbors])
                elif len(self.df_kg[x]) == 0:
                    neigh_ii.extend([0]*self.sample_ii)
                    neigh_ir.extend([0]*self.sample_ii)
                else:
                    neighbors = random.choices(self.df_kg[x], k = self.sample_ii)
                    neigh_ii.extend([nei for nei, _ in neighbors])
                    neigh_ir.extend([nei for  _, nei in neighbors])

                if len(self.df_itemuser[x])>= self.sample_iu:
                    neighbors = random.sample(self.df_itemuser[x], self.sample_iu)
                    for nei, rat, vot, tim in neighbors:
                        neigh_iu.append(nei)
                        neigh_iu_rat.append(rat)
                        neigh_iu_vot.append(vot)
                        neigh_iu_tim.append(tim)
                elif len(self.df_itemuser[x]) == 0:
                    neigh_iu.extend([0]*self.sample_iu)
                    neigh_iu_rat.extend([0.0]*self.sample_iu)
                    neigh_iu_vot.extend([0.0]*self.sample_iu)
                    neigh_iu_tim.extend([0.0]*self.sample_iu)
                else:
                    neighbors = random.choices(self.df_itemuser[x], k = self.sample_iu)
                    for nei, rat, vot, tim in neighbors:
                        neigh_iu.append(nei)
                        neigh_iu_rat.append(rat)
                        neigh_iu_vot.append(vot)
                        neigh_iu_tim.append(tim)

            neigh_rl = []
            neigh_rr = []
            for x in curr_r:
                if len(self.df_krelat[x])>= self.sample_r:
                    neighbors = random.sample(self.df_krelat[x], self.sample_r)
                    neigh_rl.extend([nei for nei, _ in neighbors])
                    neigh_rr.extend([nei for _, nei in neighbors])
                elif len(self.df_krelat[x]) == 0:
                    neigh_rl.extend([0]*self.sample_r)
                    neigh_rr.extend([0]*self.sample_r)
                else:
                    neighbors = random.choices(self.df_krelat[x], k = self.sample_r)
                    neigh_rl.extend([nei for nei, _ in neighbors])
                    neigh_rr.extend([nei for _, nei in neighbors])

            self.neigh.append(torch.tensor(neigh_uu).view((-1, sample_uu)))
            self.neigh.append(torch.tensor(neigh_uu_st).view((-1, sample_uu)))

            self.neigh.append(torch.tensor(neigh_ui).view((-1, sample_ui)))
            self.neigh.append(torch.tensor(neigh_ui_rat).view((-1, sample_ui)))
            self.neigh.append(torch.tensor(neigh_ui_vot).view((-1, sample_ui)))
            self.neigh.append(torch.tensor(neigh_ui_tim).view((-1, sample_ui)))

            self.neigh.append(torch.tensor(neigh_iu).view((-1, sample_iu)))
            self.neigh.append(torch.tensor(neigh_iu_rat).view((-1, sample_iu)))
            self.neigh.append(torch.tensor(neigh_iu_vot).view((-1, sample_iu)))
            self.neigh.append(torch.tensor(neigh_iu_tim).view((-1, sample_iu)))

            self.neigh.append(torch.tensor(neigh_ii).view((-1, sample_ii)))
            self.neigh.append(torch.tensor(neigh_ir).view((-1, sample_ii)))

            self.neigh.append(torch.tensor(neigh_rl).view((-1, sample_r)))
            self.neigh.append(torch.tensor(neigh_rr).view((-1, sample_r)))

            curr_u = list(set(curr_u+neigh_uu+neigh_iu))
            curr_v = list(set(curr_v+neigh_ui+neigh_ii+neigh_rl+neigh_rr))
            curr_r = list(set(curr_r+neigh_ir))

        

    def forward(self, u, v): #user item
        #try to find option for choices
        node = [] # 3 ka gap
        neigh = [] # 14 ka gap
        # print(u)
        u = u.int().tolist()
        v = v.int().tolist()
        # print(u)
        curr_u = list(set(u));
        curr_v = list(set(v));
        curr_r = []

        # print(len(curr_v))
        random.seed(3)

        for i in range(self.n_layers):
            node.append(curr_u)
            node.append(curr_v)
            node.append(curr_r)
            
            neigh_uu = []
            neigh_ui = []
            neigh_uu_st = []
            neigh_ui_rat = []
            neigh_ui_vot = []
            neigh_ui_tim = []
            for x in curr_u:
                neigh_uu.extend(self.neigh[0][x].tolist())
                neigh_uu_st.extend(self.neigh[1][x].tolist())
                neigh_ui.extend(self.neigh[2][x].tolist())
                neigh_ui_rat.extend(self.neigh[3][x].tolist())
                neigh_ui_vot.extend(self.neigh[4][x].tolist())
                neigh_ui_tim.extend(self.neigh[5][x].tolist())
            neigh_iu = []
            neigh_ii = []
            neigh_ir = []
            neigh_iu_rat = []
            neigh_iu_vot = []
            neigh_iu_tim = []    
            for x in curr_v:
                neigh_iu.extend(self.neigh[6][x].tolist())
                neigh_iu_rat.extend(self.neigh[7][x].tolist())
                neigh_iu_vot.extend(self.neigh[8][x].tolist())
                neigh_iu_tim.extend(self.neigh[9][x].tolist())
                neigh_ii.extend(self.neigh[10][x].tolist())
                neigh_ir.extend(self.neigh[11][x].tolist())
            neigh_rl = []
            neigh_rr = []
            for x in curr_r:
                neigh_rl.extend(self.neigh[12][x].tolist())
                neigh_rr.extend(self.neigh[13][x].tolist())

            neigh.append(neigh_uu)
            neigh.append(neigh_uu_st)

            neigh.append(neigh_ui)
            neigh.append(neigh_ui_rat)
            neigh.append(neigh_ui_vot)
            neigh.append(neigh_ui_tim)

            neigh.append(neigh_iu)
            neigh.append(neigh_iu_rat)
            neigh.append(neigh_iu_vot)
            neigh.append(neigh_iu_tim)

            neigh.append(neigh_ii)
            neigh.append(neigh_ir)

            neigh.append(neigh_rl)
            neigh.append(neigh_rr)

            curr_u = list(set(curr_u+neigh_uu+neigh_iu))
            curr_v = list(set(curr_v+neigh_ui+neigh_ii+neigh_rl+neigh_rr))
            curr_r = list(set(curr_r+neigh_ir))

        # print("idhar")
        # print(self.item_feat.grad)
        #layer(user_emb, item_emb, relat_emb, uu_emb, uu_st, uu_pos, ui_emb, ui_rat, ui_vot, ui_tim, iu_emb, iu_rat, iu_vot, iu_tim, ii_emb, ii_rel, rl_emb, rr_emb)\\10 embed
        for i, layer in enumerate(self.layers):
            # print(layer.Witem.weight.grad)
            # print(layer.Witem.bias.grad)
            # print(layer.Witem.weight.is_leaf)
            # print(layer.Witem.bias.is_leaf)
            curr = self.n_layers-i-1
            if i==self.n_layers-1:
                # act = leakyrelu
                act = torch.tanh
            else:
                # act = leakyrelu
                act = torch.tanh
            if self.isPos:
                usr_neigh_pos = torch.abs( self.df_position[neigh[curr*14]] - repeat_interleave(self.df_position[node[curr*3]], self.sample_uu, dim = 0))
            else:
                usr_neigh_pos = torch.zeros(len(neigh[curr*14]), 3)
            if i == 0:
                usr_feat, item_feat, rel_feat = layer(act, self.usr_feat[node[curr*3]], self.item_feat[node[curr*3+1]], self.rel_feat[node[curr*3+2]], self.usr_feat[neigh[curr*14]], torch.tensor(neigh[curr*14+1]), usr_neigh_pos, self.item_feat[neigh[curr*14+2]], torch.tensor(neigh[curr*14+3]), torch.tensor(neigh[curr*14+4]), torch.tensor(neigh[curr*14+5]), self.usr_feat[neigh[curr*14+6]], torch.tensor(neigh[curr*14+7]), torch.tensor(neigh[curr*14+8]), torch.tensor(neigh[curr*14+9]), self.item_feat[neigh[curr*14+10]], self.rel_feat[neigh[curr*14+11]], self.item_feat[neigh[curr*14+12]], self.item_feat[neigh[curr*14+13]])
            else:
                user_encoder = LabelEncoder()
                entity_encoder = LabelEncoder()
                relation_encoder = LabelEncoder()
                user_encoder.fit(node[(curr+1)*3])
                entity_encoder.fit(node[(curr+1)*3+1])
                relation_encoder.fit(node[(curr+1)*3+2])
                usr_feat, item_feat, rel_feat = layer(act, usr_feat[user_encoder.transform(node[curr*3])], item_feat[entity_encoder.transform(node[curr*3+1])], rel_feat[relation_encoder.transform(node[curr*3+2])], usr_feat[user_encoder.transform(neigh[curr*14])], torch.tensor(neigh[curr*14+1]), usr_neigh_pos, item_feat[entity_encoder.transform(neigh[curr*14+2])], torch.tensor(neigh[curr*14+3]), torch.tensor(neigh[curr*14+4]), torch.tensor(neigh[curr*14+5]), usr_feat[user_encoder.transform(neigh[curr*14+6])], torch.tensor(neigh[curr*14+7]), torch.tensor(neigh[curr*14+8]), torch.tensor(neigh[curr*14+9]), item_feat[entity_encoder.transform(neigh[curr*14+10])], rel_feat[relation_encoder.transform(neigh[curr*14+11])], item_feat[entity_encoder.transform(neigh[curr*14+12])], item_feat[entity_encoder.transform(neigh[curr*14+13])])
            # x = torch.norm(usr_feat, dim=1)
            # x[x==0]=1.0
            # usr_feat = torch.transpose(torch.div(torch.transpose(usr_feat, 0,1) , x), 0, 1)
            # x = torch.norm(item_feat, dim=1)
            # x[x==0]=1.0
            # item_feat = torch.transpose(torch.div(torch.transpose(item_feat, 0,1) , x), 0, 1)
            # x = torch.norm(rel_feat, dim=1)
            # x[x==0]=1.0
            # rel_feat = torch.transpose(torch.div(torch.transpose(rel_feat, 0,1) , x), 0, 1)
            # usr_feat, item_feat, rel_feat = layer(usr_feat, item_feat,rel_feat)
        # print(usr_feat.requires_grad)
        user_encoder = LabelEncoder()
        entity_encoder = LabelEncoder()
        user_encoder.fit(u)
        entity_encoder.fit(v)
        u = user_encoder.transform(u)
        v = entity_encoder.transform(v)
        usr_feat = usr_feat[u]
        item_feat = item_feat[v]
        # print(usr_feat)
        # print(item_feat)
        scores = (usr_feat * item_feat).sum(dim = 1)
        # scores = (scores + 1.0)/2.0
        # scores = torch.sigmoid(scores)
        
        return torch.sigmoid(scores)
    