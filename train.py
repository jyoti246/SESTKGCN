import pandas as pd
import numpy as np
import argparse
import random
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='music', help='which dataset to use')
parser.add_argument('--n_epochs', type=int, default=100, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=32, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=32, help='dimension of user and entity embeddings')
parser.add_argument('--layers', type=int, default=1, help='layers in model')
parser.add_argument('--n_iter', type=int, default=100, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
parser.add_argument('--l2_weight', type=float, default=2e-4, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=2e-2, help='learning rate')
parser.add_argument('--ratio', type=float, default=0.8, help='size of training dataset')
parser.add_argument('--seed', type=float, default=10, help='seed for model initialization')
parser.add_argument('--is_emb', type=float, default=1, help='for model initialization')
parser.add_argument('--agg', type=str, default="concat", help='for model aggregator initialization')
args = parser.parse_args(['--l2_weight', '2e-3'])


def process_movie(cutoff):
    df_ratings = pd.read_csv('data/movie/ratings.csv', sep=',',names=['userID', 'itemID', 'rating', 'timestamp'], skiprows=1)
    df_item2id = pd.read_csv('data/movie/item_index2entity_id.txt', sep='\t', header=None, names=['item','id'])
    df_kgs = pd.read_csv('data/movie/kg.txt', sep='\t', header=None, names=['head','relation','tail'])
        
    
    # df_ratings = df_ratings[df_ratings['itemID'].isin(df_tags['itemID'])]
    df_ratings = df_ratings[df_ratings['itemID'].isin(df_item2id['item'])]
    # df_ratings.reset_index(inplace=True, drop=True)

    # df_ratings = df_ratings.groupby('userID')['userID', 'itemID', 'rating', 'timestamp'].filter(lambda x: len(x) >= 3) 
    df_ratings.reset_index(inplace=True, drop=True)
    df_ratings = df_ratings[:2000000]
    

    # df_tags = df_tags[df_tags['itemID'].isin(df_item2id['item'])]
    # df_tags.reset_index(inplace=True, drop=True)

    user_encoder = LabelEncoder()
    entity_encoder = LabelEncoder()
    relation_encoder = LabelEncoder()

    user_encoder.fit(df_ratings['userID'])
    entity_encoder.fit(pd.concat([df_item2id['id'], df_kgs['head'], df_kgs['tail']]))
    relation_encoder.fit(df_kgs['relation'])
    
    item2id_dict = dict(zip(df_item2id['item'], df_item2id['id']))
    df_ratings['itemID'] = df_ratings['itemID'].apply(lambda x: item2id_dict[x])
    
    df_ratings['itemID'] = entity_encoder.transform(df_ratings['itemID'])
    df_ratings['userID'] = user_encoder.transform(df_ratings['userID'])
    df_kgs['relation'] = relation_encoder.transform(df_kgs['relation'])
    df_kgs['head'] = entity_encoder.transform(df_kgs['head'])
    df_kgs['tail'] = entity_encoder.transform(df_kgs['tail'])
    n_user = len(user_encoder.classes_)
    n_item = len(entity_encoder.classes_)
    n_relation = len(relation_encoder.classes_)
    df_position = torch.zeros(1,3)
    df_kg = [[] for x in range(0,n_item)]
    df_sg = [[] for x in range(0,n_user)]
    df_krelat = [[] for x in range(0,n_relation)]
    for ind in df_kgs.index:
        df_kg[df_kgs['head'][ind]].append((df_kgs['tail'][ind], df_kgs['relation'][ind]));
        df_krelat[df_kgs['relation'][ind]].append((df_kgs['head'][ind], df_kgs['tail'][ind]))
        df_kg[df_kgs['tail'][ind]].append((df_kgs['head'][ind], df_kgs['relation'][ind]));
        df_krelat[df_kgs['relation'][ind]].append((df_kgs['tail'][ind], df_kgs['head'][ind]))

    max_rating = 5.0
    
    std_tim = df_ratings['timestamp'].std()
    men_tim = df_ratings['timestamp'].mean()

    df_ratings['timestamp'] = (df_ratings['timestamp']- men_tim)/std_tim 
    df_ratings['rating'] = df_ratings['rating']/ max_rating
    df_ratings['rating'] = df_ratings['rating'].apply(lambda x: 0 if x < .7 else x)

    df_max = df_ratings.groupby('userID')['itemID'].max()
    df_ratings['maxRat'] = df_ratings.userID.map(lambda x: df_max[x] ) 
    df_test = df_ratings[df_ratings['itemID'] == df_ratings['maxRat']]
    df_test = df_test.drop_duplicates(subset='userID', keep="last")
    df_test.drop('maxRat',axis='columns', inplace=True)
    df_ratings.drop('maxRat',axis='columns', inplace=True)
    df_ratings = df_ratings.drop(df_test.index)

    
    df_test.reset_index(inplace=True, drop=True)
    df_ratings.reset_index(inplace=True, drop=True)
    

    df_useritem = [[] for x in range(n_user)]
    df_itemuser = [[] for x in range(n_item)]
    for ind in df_ratings.index:
        if df_ratings['rating'][ind] > 0:
            df_useritem[df_ratings['userID'][ind]].append((df_ratings['itemID'][ind], df_ratings['rating'][ind], 1, df_ratings['timestamp'][ind]))
            df_itemuser[df_ratings['itemID'][ind]].append((df_ratings['userID'][ind], df_ratings['rating'][ind], 1, df_ratings['timestamp'][ind]))
    df_test.drop('timestamp',axis='columns', inplace=True)
    df_ratings.drop('timestamp',axis='columns', inplace=True)

    df_ratings['rating'] = df_ratings['rating'].apply(lambda x: 0 if x < .8 else 1)
    df_test['rating'] = df_test['rating'].apply(lambda x: 0 if x < .8 else 1)
    df_ratings.reset_index(inplace=True, drop=True)
    df_ratings = df_ratings[df_ratings['rating']==1.0]
    df_test = df_test[df_test['rating']==1.0]
    return df_ratings, df_test, df_sg, df_kg, df_useritem, df_itemuser, df_position, df_krelat, n_user, n_item, n_relation, max_rating, 0

def process_movie_change(cutoff):
    df_ratings = pd.read_csv('data/movie/ratings.csv', sep=',',names=['userID', 'itemID', 'rating', 'timestamp'], skiprows=1)
    df_tags = pd.read_csv('data/movie/genome-scores.csv', sep=',',names=['itemID', 'tagID', 'relevance'], skiprows=1)
    df_item2id = pd.read_csv('data/movie/item_index2entity_id.txt', sep='\t', header=None, names=['item','id'])
    df_tags = df_tags[df_tags['relevance']>=cutoff]

    
    df_ratings = df_ratings[df_ratings['itemID'].isin(df_tags['itemID'])]
    df_ratings = df_ratings[df_ratings['itemID'].isin(df_item2id['item'])]
    df_ratings.reset_index(inplace=True, drop=True)
    df_ratings = df_ratings[:2000000]
    
    # df_ratings = df_ratings.groupby('userID')['userID', 'itemID', 'rating', 'timestamp'].filter(lambda x: len(x) >= 5) 
    # df_ratings.reset_index(inplace=True, drop=True)


    df_tags = df_tags[df_tags['itemID'].isin(df_item2id['item'])]
    df_tags.reset_index(inplace=True, drop=True)

    user_encoder = LabelEncoder()
    entity_encoder = LabelEncoder()
    relation_encoder = LabelEncoder()

    user_encoder.fit(df_ratings['userID'])
    entity_encoder.fit(pd.concat([df_ratings['itemID'], df_tags['itemID']]))
    relation_encoder.fit(df_tags['tagID'])
    
    df_ratings['itemID'] = entity_encoder.transform(df_ratings['itemID'])
    df_tags['itemID'] = entity_encoder.transform(df_tags['itemID'])
    df_tags['tagID'] = relation_encoder.transform(df_tags['tagID'])
    df_ratings['userID'] = user_encoder.transform(df_ratings['userID'])

    n_user = len(user_encoder.classes_)
    n_item = len(entity_encoder.classes_)
    n_relation = len(relation_encoder.classes_)
    df_position = torch.zeros(1,3)
    df_kg = [[] for x in range(0,n_item)]
    df_sg = [[] for x in range(0,n_user)]
    df_krelat = [[] for x in range(0,n_relation)]
    for tag, group in df_tags.groupby(['tagID']):
        for i in group['itemID']:
            for j in group['itemID']:
                if i!=j:
                    df_kg[i].append((j, tag));
                    df_krelat[tag].append((i, j))

    max_rating = 5.0
    
    std_tim = df_ratings['timestamp'].std()
    men_tim = df_ratings['timestamp'].mean()

    df_ratings['timestamp'] = (df_ratings['timestamp']- men_tim)/std_tim 
    df_ratings['rating'] = df_ratings['rating']/ max_rating
    df_ratings['rating'] = df_ratings['rating'].apply(lambda x: 0 if x < .7 else x)

    # df_max = df_ratings.groupby('userID')['itemID'].max()
    # df_ratings['maxRat'] = df_ratings.userID.map(lambda x: df_max[x] ) 
    # df_test = df_ratings[df_ratings['itemID'] == df_ratings['maxRat']]
    # df_test = df_test.drop_duplicates(subset='userID', keep="last")
    # df_test.drop('maxRat',axis='columns', inplace=True)
    # df_ratings.drop('maxRat',axis='columns', inplace=True)
    # df_ratings = df_ratings.drop(df_test.index)

    df_test = df_ratings.drop_duplicates(subset='userID', keep="last")
    df_ratings = df_ratings.drop(df_test.index)

    df_test.reset_index(inplace=True, drop=True)
    # df_validate.reset_index(inplace=True, drop=True)
    df_ratings.reset_index(inplace=True, drop=True)
    

    df_useritem = [[] for x in range(n_user)]
    df_itemuser = [[] for x in range(n_item)]
    for ind in df_ratings.index:
        if df_ratings['rating'][ind] > 0:
            df_useritem[df_ratings['userID'][ind]].append((df_ratings['itemID'][ind], df_ratings['rating'][ind], 1, df_ratings['timestamp'][ind]))
            df_itemuser[df_ratings['itemID'][ind]].append((df_ratings['userID'][ind], df_ratings['rating'][ind], 1, df_ratings['timestamp'][ind]))
    df_test.drop('timestamp',axis='columns', inplace=True)
    # df_validate.drop('timestamp',axis='columns', inplace=True)
    df_ratings.drop('timestamp',axis='columns', inplace=True)

    df_ratings['rating'] = df_ratings['rating'].apply(lambda x: 0 if x < .8 else 1)
    df_test['rating'] = df_test['rating'].apply(lambda x: 0 if x < .8 else 1)
    df_ratings = df_ratings[df_ratings['rating']==1.0]
    df_test = df_test[df_test['rating']==1.0]
    return df_ratings, df_test, df_sg, df_kg, df_useritem, df_itemuser, df_position, df_krelat, n_user, n_item, n_relation, max_rating, 0
    

def gen_kg_music():
    df = pd.read_csv('data/last_fm/user_taggedartists-timestamps.dat', sep='\t',names=['userID', 'itemID', 'tagID', 'timestamp'], skiprows=1)[['itemID', 'tagID']]
    dups = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0:'count'})
    dfn = dups[['count']]
    dupsn = dfn.groupby(dfn.columns.tolist()).size().reset_index().rename(columns={0:'countn'})
    # dups.to_csv('data/last_fm/item_tag.dat',index=False)
    print(dupsn)
    print(dups)


def process_music(cutoff):
    df_ratings = pd.read_csv('data/last_fm/user_artists.dat', sep='\t',names=['userID', 'itemID', 'rating'], skiprows=1)
    df_tags = pd.read_csv('data/last_fm/item_tag.dat', sep=',',names=['itemID', 'tagID', 'relevance'], skiprows=1)
    df_socio = pd.read_csv('data/last_fm/user_friends.dat', sep='\t', names=['userID','friendID'], skiprows=1)

    df_tags = df_tags[df_tags['relevance']>=cutoff]
    # print(df_ratings)
    df_ratings = df_ratings[df_ratings['itemID'].isin(df_tags['itemID'])]
    
    # print(df_ratings)
    df_ratings = df_ratings.groupby('userID')['userID', 'itemID', 'rating'].filter(lambda x: len(x) >= 3) 
    df_ratings.reset_index(inplace=True, drop=True)
    df_tags.reset_index(inplace=True, drop=True)

    user_encoder = LabelEncoder()
    entity_encoder = LabelEncoder()
    relation_encoder = LabelEncoder()

    user_encoder.fit(pd.concat([df_ratings['userID'], df_socio['friendID'],df_socio['userID'] ]))
    # df_item2id['id'] and df_kg[['head', 'tail']] represents new entity ID
    entity_encoder.fit(pd.concat([df_ratings['itemID'], df_tags['itemID']]))
    relation_encoder.fit(df_tags['tagID'])
    
    df_ratings['itemID'] = entity_encoder.transform(df_ratings['itemID'])
    df_tags['itemID'] = entity_encoder.transform(df_tags['itemID'])
    df_tags['tagID'] = relation_encoder.transform(df_tags['tagID'])
    df_ratings['userID'] = user_encoder.transform(df_ratings['userID'])
    df_socio['friendID'] = user_encoder.transform(df_socio['friendID'])
    df_socio['userID'] = user_encoder.transform(df_socio['userID'])

    n_user = len(user_encoder.classes_)
    n_item = len(entity_encoder.classes_)
    n_relation = len(relation_encoder.classes_)
    df_position = torch.zeros(1,3)
    df_kg = [[] for x in range(0,n_item)]
    df_sg = [[] for x in range(0,n_user)]
    df_krelat = [[] for x in range(0,n_relation)]
    for tag, group in df_tags.groupby(['tagID']):
        for i in group['itemID']:
            for j in group['itemID']:
                if i!=j:
                    df_kg[i].append((j, tag));
                    df_krelat[tag].append((i, j))

    for ind in df_socio.index:
        df_sg[df_socio['userID'][ind]].append((df_socio['friendID'][ind],1.0))
    max_rating =  df_ratings['rating'].max()
    # std_pos = np.std(np.array(list(user_position.values())), axis=0)
    # men_pos = np.mean(np.array(list(user_position.values())), axis=0)
    # user_position = {k: [(a-b)/c if c!=0 else a-b for a,b,c in zip(v, men_pos, std_pos) ]  for k, v in user_position.items()}
    
    df_ratings['rating'] = df_ratings['rating']/ max_rating
    # take last timestamped item in test set and also incorporate timestamp
    df_max = df_ratings.groupby('userID')['itemID'].max()
    df_ratings['maxRat'] = df_ratings.userID.map(lambda x: df_max[x] ) 
    df_test = df_ratings[df_ratings['itemID'] == df_ratings['maxRat']]
    df_test = df_test.drop_duplicates(subset='userID', keep="last")
    df_test.drop('maxRat',axis='columns', inplace=True)
    df_ratings.drop('maxRat',axis='columns', inplace=True)
    df_ratings = df_ratings.drop(df_test.index)
    # again take timestamp
    # df_validate = df_ratings.drop_duplicates(subset='userID', keep="last")
    # df_ratings = df_ratings.drop(df_validate.index)

    df_test.reset_index(inplace=True, drop=True)
    # df_validate.reset_index(inplace=True, drop=True)
    df_ratings.reset_index(inplace=True, drop=True)

    df_useritem = [[] for x in range(n_user)]
    df_itemuser = [[] for x in range(n_item)]
    for ind in df_ratings.index:
        df_useritem[df_ratings['userID'][ind]].append((df_ratings['itemID'][ind], df_ratings['rating'][ind], 1, 1))
        df_itemuser[df_ratings['itemID'][ind]].append((df_ratings['userID'][ind], df_ratings['rating'][ind], 1, 1))
    df_ratings['rating'] = 1.0
    # df_validate['rating'] = 1.0
    df_test['rating'] = 1.0
    
    return df_ratings, df_test,df_sg, df_kg, df_useritem, df_itemuser, df_position, df_krelat, n_user, n_item, n_relation, max_rating, 0
    
import math

def process_book(age_imp):
    df_ratings = pd.read_csv('data/books/book_rating.dat', names=['userID', 'itemID', 'rating'], skiprows=1)
    df_tags = pd.read_csv('data/books/BX-Books.csv', sep=';',names=['itemID', 'title', 'author', 'a','b','c','d','e'], skiprows=1, usecols = ['itemID', 'title', 'author'],  encoding= 'unicode_escape')
    df_socio = pd.read_csv('data/books/BX-Users.csv', sep=';', names=['userID','place','age'], skiprows=1, encoding= 'unicode_escape')
    df_socio = df_socio[df_socio['userID'].isin(df_ratings['userID'])]
    

    user_encoder = LabelEncoder()
    entity_encoder = LabelEncoder()
    relation_encoder = LabelEncoder()

    user_encoder.fit(pd.concat([df_ratings['userID'], df_socio['userID']]))
    df_ratings['userID'] = user_encoder.transform(df_ratings['userID'])
    df_socio['userID'] = user_encoder.transform(df_socio['userID'])
    df_ratings = df_ratings[df_ratings['userID']<3000]
    df_socio = df_socio[df_socio['userID']<3000]
    df_tags = df_tags[df_tags['itemID'].isin(df_ratings['itemID'])]
    
    entity_encoder.fit(df_ratings['itemID'])
    df_ratings['itemID'] = entity_encoder.transform(df_ratings['itemID'])
    df_tags['itemID'] = entity_encoder.transform(df_tags['itemID'])

    relation_encoder.fit(pd.concat([df_tags['title'], df_tags['author']]))
    df_tags['title'] = relation_encoder.transform(df_tags['title'])
    df_tags['author'] = relation_encoder.transform(df_tags['author'])
    

    n_user = 3000
    n_item = len(entity_encoder.classes_)
    n_relation = 2  #len(relation_encoder.classes_)

    df_position = torch.zeros(1,3)
    df_kg = [[] for x in range(0,n_item)]
    df_sg = [[] for x in range(0,n_user)]
    df_krelat = [[] for x in range(0,n_relation)]

    df_tags.reset_index(inplace=True, drop=True)
    for tag, group in df_tags.groupby(['title']):
        for i in group['itemID']:
            for j in group['itemID']:
                if i!=j:
                    df_kg[i].append((j, 0));
                    df_krelat[0].append((i, j))
    for tag, group in df_tags.groupby(['author']):
        for i in group['itemID']:
            for j in group['itemID']:
                if i!=j:
                    df_kg[i].append((j, 1));
                    df_krelat[1].append((i, j))
    
    df_socio.reset_index(inplace=True, drop=True)
    for i in df_socio.index:
        for j in df_socio.index:
            if i != j:
                tag = 0.0
                if df_socio['place'][i].split(sep = ', ', maxsplit = 1)[1] == df_socio['place'][j].split(sep = ', ', maxsplit = 1)[1]:
                    tag += 1
                if math.isnan(df_socio['age'][j])==False and math.isnan(df_socio['age'][i])==False and abs(df_socio['age'][i] - df_socio['age'][j]) == 5:
                    tag += age_imp
                df_sg[df_socio['userID'][i]].append((df_socio['userID'][j],tag))

        
    max_rating =  10.0
    
    df_ratings['rating'] = df_ratings['rating']/ max_rating
    df_max = df_ratings.groupby('userID')['itemID'].max()
    df_ratings['maxRat'] = df_ratings.userID.map(lambda x: df_max[x] ) 
    df_test = df_ratings[df_ratings['itemID'] == df_ratings['maxRat']]
    df_test = df_test.drop_duplicates(subset='userID', keep="last")
    df_test.drop('maxRat',axis='columns', inplace=True)
    df_ratings.drop('maxRat',axis='columns', inplace=True)
    df_ratings = df_ratings.drop(df_test.index)
    # df_validate = df_ratings.drop_duplicates(subset='userID', keep="last")
    # df_ratings = df_ratings.drop(df_validate.index)

    df_test.reset_index(inplace=True, drop=True)
    # df_validate.reset_index(inplace=True, drop=True)
    df_ratings.reset_index(inplace=True, drop=True)

    df_useritem = [[] for x in range(n_user)]
    df_itemuser = [[] for x in range(n_item)]
    for ind in df_ratings.index:
        df_useritem[df_ratings['userID'][ind]].append((df_ratings['itemID'][ind], df_ratings['rating'][ind], 1, 1))
        df_itemuser[df_ratings['itemID'][ind]].append((df_ratings['userID'][ind], df_ratings['rating'][ind], 1, 1))
    
    df_ratings['rating'] = df_ratings['rating'].apply(lambda x: 0 if x < .2 else 1)
    # df_validate['rating'] = df_validate['rating'].apply(lambda x: 0 if x < .2 else 1)
    df_test['rating'] = df_test['rating'].apply(lambda x: 0 if x < .2 else 1)
    df_ratings = df_ratings[df_ratings['rating']==1.0]
    df_test = df_test[df_test['rating']==1.0]
    return df_ratings, df_test, df_sg, df_kg, df_useritem, df_itemuser, df_position, df_krelat, n_user, n_item, n_relation, max_rating, 0
    


if args.dataset == 'music':
	# gen_kg_music()
	df_ratings, df_test,df_sg, df_kg, df_useritem, df_itemuser, df_position, df_krelat, n_user, n_item, n_relation, max_rating, isPos = process_music(3)
elif args.dataset == 'movie':
	df_ratings, df_test, df_sg, df_kg, df_useritem, df_itemuser, df_position, df_krelat, n_user, n_item, n_relation, max_rating, isPos = process_movie(0.9)
else:
    df_ratings, df_test, df_sg, df_kg, df_useritem, df_itemuser, df_position, df_krelat, n_user, n_item, n_relation, max_rating, isPos = process_book(3)


print(n_user, n_item)

# print(df_ratings)
# negative sample not purchased by neighbour
def negative_sample(ratings, n_item, test):
    full_item_set = set(range(n_item))
    user_list = []
    item_list = []
    label_list = []
    random.seed(100)
    df_tst = test.set_index('userID')
    for user, group in ratings.groupby(['userID']):
        item_set = set(group['itemID'])
        if user in df_tst.index:
            item_set.add(df_tst['itemID'][user])
            negative_set = full_item_set - item_set
            negative_sampled = random.sample(negative_set, min(max(0,len(item_set)-1),len(negative_set)))
        else:
            negative_set = full_item_set - item_set
            negative_sampled = random.sample(negative_set, min(len(item_set), len(negative_set))) 
        user_list.extend([user] * len(negative_sampled))
        item_list.extend(negative_sampled)
        label_list.extend([0.0] * len(negative_sampled))
    negative = pd.DataFrame({'userID': user_list, 'itemID': item_list, 'rating': label_list})
    print(negative)
    ratings = pd.concat([ratings, negative])
    return ratings

df_dataset = negative_sample(df_ratings, n_item, df_test)
df_dataset = df_dataset.sample(frac=1, replace=False, random_state=999)


# Dataset class
class KGCNDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        user_id = np.array(self.df.iloc[idx]['userID'])
        item_id = np.array(self.df.iloc[idx]['itemID'])
        label = np.array(self.df.iloc[idx]['rating'], dtype=np.float32)
        return user_id, item_id, label


x_train, x_test, y_train, y_test = train_test_split(df_dataset, df_dataset['rating'], test_size=1 - args.ratio, shuffle=True, random_state=999)
print(x_train,y_train)
train_dataset = KGCNDataset(x_train)
test_dataset = KGCNDataset(x_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

print(n_user, n_item, n_relation)

# prepare network, loss function, optimizer
device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
sample_uu= args.neighbor_sample_size
sample_ui=args.neighbor_sample_size
sample_iu = args.neighbor_sample_size
sample_ii= args.neighbor_sample_size
sample_r= args.neighbor_sample_size
dim = args.dim
n_layers = args.layers
torch.manual_seed(args.seed)
net = SESTKGCN(df_position, df_sg, df_useritem, df_itemuser, df_kg, df_krelat, n_user, n_item, n_relation, dim, n_layers, max_rating,  1, 1, 1, isPos, sample_uu, sample_ui, sample_iu, sample_ii, sample_r, device, args.agg).to(device)
# net.load_state_dict(torch.load("/content/drive/My Drive/btp_data/data/last_fm/weights_only.pth"))
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2_weight)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 50, gamma=0.5)
print('device: ', device)
print(sum(p.numel() for p in net.parameters() if p.requires_grad))
random.seed(3)
for p in net.parameters():
    if p.requires_grad:
         print(p.name, p.data.size())


# train
loss_list = []
test_loss_list = []
auc_score_list = []
prec_list = []
rec_list = []
f1_list = []
for epoch in range(args.n_epochs):
    running_loss = 0.0
    total_items = 0.0
    net.train()
    tot_f1=0.0
    for i, (user_ids, item_ids, labels) in enumerate(train_loader):
        user_ids, item_ids, labels = user_ids.double().to(device), item_ids.double().to(device), labels.double().to(device)
        
        # print(user_ids, item_ids)
        optimizer.zero_grad()
        outputs = net(user_ids, item_ids)
        if i==0:
          print(labels[:10])
          print(outputs[:10])
        # outputs.requires_grad = True
        
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        # loss /= labels.size()[0]
        # loss = torch.sqrt(criterion(x, y))
        total_items += labels.size()[0]
        # print(labels.size()[0])
        # print(labels.requires_grad)
        # print(outputs.requires_grad)
        loss.backward()
        # print(net.)
        optimizer.step()
    
    # print train loss per every epoch
    print('[Epoch {}]train_loss: '.format(epoch+1), running_loss / len(train_loader))
    loss_list.append(running_loss / len(train_loader))
    
        
    # evaluate per every epoch
    total_items = 0.0
    net.eval()
    with torch.no_grad():
        test_loss = 0
        total_roc = 0
        total_prec = 0
        total_recall = 0
        total_f1 = 0
        for user_ids, item_ids, labels in test_loader:
            user_ids, item_ids, labels = user_ids.double().to(device), item_ids.double().to(device), labels.double().to(device)
            outputs = net(user_ids, item_ids)
            test_loss += criterion(outputs, labels).item()
            total_items += labels.size()[0]
            total_roc += roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
            prec, rec, f1, _ = precision_recall_fscore_support(labels >= .5, outputs >= .5, average='binary', zero_division=1)
            total_prec += prec
            total_recall += rec
            total_f1 += f1
        print('[Epoch {}]test_loss: '.format(epoch+1), test_loss / len(test_loader))
        print('Test Precision: {:.4f} | Recall: {:.4f} | F1: {:.4f}'.format(total_prec/ len(test_loader), total_recall/ len(test_loader), total_f1/ len(test_loader)))
        prec_list.append(total_prec/ len(test_loader))
        rec_list.append(total_recall/ len(test_loader))
        f1_list.append(total_f1/ len(test_loader))
        test_loss_list.append(test_loss / len(test_loader))
        auc_score_list.append(total_roc / len(test_loader))


# plot losses / scores
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,4))  # 1 row, 2 columns
ax1.plot(loss_list, label='train loss')
ax1.plot(test_loss_list, label='test loss')
ax2.plot(auc_score_list, label = 'auc')
ax2.plot(prec_list, label = 'prec')
ax2.plot(rec_list, label = 'rec')
ax2.plot(f1_list, label = 'f1')
ax1.legend()
ax2.legend()
plt.tight_layout()

# schedulr = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma=4)
# schedulr.step()

# for param_group in optimizer.param_groups:
#     print(param_group['lr'])

df_tst = df_test.set_index('userID')
hit = 0.0
tot = 0.0
for usr, group in df_ratings.groupby(['userID']):
    if usr not in df_tst.index:
      continue
    item_ids = np.arange(start=0, stop=n_item, step=1)
    item_ids = np.delete(item_ids, group['itemID'])
    item_ids = torch.from_numpy(item_ids)
    # user_ids = np.full(item_ids.shape, usr, dtype=int)
    user_ids = torch.empty(item_ids.size()[0]).fill_(usr)
    # print(user_ids, item_ids)
    net.eval()
    outputs = net(user_ids, item_ids)
    ratings, indices = torch.sort(outputs, descending=True)
    # print(ratings)
    indices = indices[:10]
    # print(indices)
    if df_tst['itemID'][usr] in indices:
        hit += 1.0
    print(hit) 
    tot += 1.0
print(hit, tot, hit/tot)