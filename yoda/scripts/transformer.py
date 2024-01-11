import pandas as pd
import os, gc
import numpy as np
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

##############################################################
# Fix fastai bug to enable fp16 training with dictionaries
##############################################################
import torch
from fastai.vision.all import *
def flatten(o):
    "Concatenate all collections and items as a generator"
    for item in o:
        if isinstance(o, dict): yield o[item]; continue
        elif isinstance(item, str): yield item; continue
        try: yield from flatten(item)
        except TypeError: yield item

from torch.cuda.amp import GradScaler, autocast
@delegates(GradScaler)
class MixedPrecision(Callback):
    "Mixed precision training using Pytorch's `autocast` and `GradScaler`"
    order = 10
    def __init__(self, **kwargs): self.kwargs = kwargs
    def before_fit(self):
        self.autocast,self.learn.scaler,self.scales = autocast(),GradScaler(**self.kwargs),L()
    def before_batch(self): self.autocast.__enter__()
    def after_pred(self):
        if next(flatten(self.pred)).dtype==torch.float16: self.learn.pred = to_float(self.pred)
    def after_loss(self): self.autocast.__exit__(None, None, None)
    def before_backward(self): self.learn.loss_grad = self.scaler.scale(self.loss_grad)
    def before_step(self):
        "Use `self` as a fake optimizer. `self.skipped` will be set to True `after_step` if gradients overflow. "
        self.skipped=True
        self.scaler.step(self)
        if self.skipped: raise CancelStepException()
        self.scales.append(self.scaler.get_scale())
    def after_step(self): self.learn.scaler.update()

    @property
    def param_groups(self):
        "Pretend to be an optimizer for `GradScaler`"
        return self.opt.param_groups
    def step(self, *args, **kwargs):
        "Fake optimizer step to detect whether this batch was skipped from `GradScaler`"
        self.skipped=False
    def after_fit(self): self.autocast,self.learn.scaler,self.scales = None,None,None

import fastai
fastai.callback.fp16.MixedPrecision = MixedPrecision



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def dict_to(x, device):
    return {k:x[k].to(device) for k in x}

def to_device(x, device):
    return tuple(dict_to(e,device) for e in x)

class DeviceDataLoader:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for batch in self.dataloader:
            yield tuple(dict_to(x, self.device) for x in batch)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=16, M=10000):
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[...,None] * emb[None,...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb



# class LenMatchBatchSampler(torch.utils.data.BatchSampler):
#     def __iter__(self):
#         buckets = [[]] * 100
#         yielded = 0

#         for idx in self.sampler:
#             s = self.sampler.data_source[idx]
#             if isinstance(s,tuple): L = s[0]["mask"].sum()
#             else: L = s["mask"].sum()
#             L = max(1,L // 16)
#             if len(buckets[L]) == 0:  buckets[L] = []
#             buckets[L].append(idx)

#             if len(buckets[L]) == self.batch_size:
#                 batch = list(buckets[L])
#                 yield batch
#                 yielded += 1
#                 buckets[L] = []
#         batch = []
#         leftover = [idx for bucket in buckets for idx in bucket]
#         for idx in leftover:
#             batch.append(idx)
#             if len(batch) == self.batch_size:
#                 yielded += 1
#                 yield batch
#                 batch = []
#         if len(batch) > 0 and not self.drop_last:
#             yielded += 1
#             yield batch




###############################
#  IMPLEMENTATION STARTS HERE
#################################

from ubergauss.optimization import groupedCV
class RNA_Dataset(Dataset):
    def __init__(self, df, mode='train', seed=2023, fold=0, nfolds=4, **kwargs):
        self.seq_map = {'A':0,'C':1,'G':2,'U':3} # TODO consider '-' gap encoding
        self.Lmax = 400

        '''
        there is no train test split anymore.
        for the test set we just censor the set labels
        '''
        labels = df['set'].values

        # split = list(groupedCV(n_splits=nfolds, randseed=seed).split(df,labels, groups=labels))[fold][0 if mode=='train' else 1]
        train_set, test_set = list(groupedCV(n_splits=nfolds, randseed=seed).split(df,labels, groups=labels))[fold]

        self.trainlabels = df['set'].values
        self.trainlabels[train_set] = -1
        # df = df.iloc[split].sample(frac=1).reset_index(drop=True)

        if mode == 'eval':
            df = df.iloc[test_set].reset_index(drop=True)

        # our df has 'set' (labels), sequence, pos1id, pos2id

        self.seq = df['sequence'].values
        self.testlabels = df['set'].values
        self.p1 = df['pos1id'].values
        self.p2 = df['pos2id'].values
        self[0]

    def __len__(self):
        return len(self.seq)


    def getseq(self, idx):
        seq = self.seq[idx]
        seq = [self.seq_map[s] for s in seq]
        n_nuc = len(seq)
        seq = np.eye(4)[seq]
        seq = np.hstack((seq.T, np.zeros((4,self.Lmax-n_nuc))))
        return n_nuc, seq.T

    def __getitem__(self, idx):
        # label = np.array(self.trainlabels[idx])

        n_nuc, seq = self.getseq(idx)

        # po1 = np.full(self.Lmax, False)
        # po1[list(self.p1[idx])] = True
        # po2 = np.full(self.Lmax, False)
        # po2[list(self.p2[idx])] = True

        po1 = self.p1[idx]
        po2 = self.p2[idx]

        p1 = torch.full((self.Lmax,),-1, dtype=torch.long)
        p2 = torch.full((self.Lmax,),-1, dtype = torch.long)

        # p1 = torch.zeros(self.Lmax, dtype=torch.long)
        # p2 = torch.zeros(self.Lmax, dtype=torch.long)


        p1[:len(po1)] = torch.tensor(po1)
        p2[:len(po2)] = torch.tensor(po2)
        return {'seq':torch.from_numpy(seq),'len':n_nuc , 'p1':p1, 'p2':p2},\
                    {'testlabel':self.testlabels[idx], 'trainlabel':self.trainlabels[idx]}

import structout as so
def hm(mat, **kw):
    print(mat.shape)
    so.heatmap(mat.cpu().numpy(),**kw)


class RNA_Model(nn.Module):
    def __init__(self, dim=128, depth=1, head_size=32, **kwargs): # depth was 12 and headsize was 32
        super().__init__()
        # self.emb = nn.Embedding(4,dim-64)
        self.pos_enc = SinusoidalPosEmb(32)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=dim//head_size, dim_feedforward=4*dim,
                dropout=0.15, activation=nn.GELU(), batch_first=True, norm_first=True), depth) # dropput was .1
        self.fc1 = nn.Linear(400*dim, 1024)
        self.fc2 = nn.Linear(1024, 8)

    def forward(self, x0):
        # note i removed the masking maybe that was too much cleaning
        seq = x0['seq']
        p1,p2 = x0['p1'], x0['p2']
        batch_size = seq.shape[0]

        pos = torch.arange(400, device=seq.device)
        pos= pos.unsqueeze(0)
        pos = self.pos_enc(pos) # bsx400
        pos = torch.repeat_interleave(pos, batch_size, dim=0 )

        struct = torch.zeros((batch_size,400,32), device = pos.device)

        p1 = torch.where(p1>=  0)
        p2 = torch.where(p2>=  0)
        breakpoint()
        struct[p1] = pos[p2]
        struct[p2] = pos[p1]

        # indi = p1.unsqueeze(-1)
        # struct.scatter_(2, indi, pos.gather(2, indi))
        # indi = p2.unsqueeze(-1)
        # struct.scatter_(2, indi, pos.gather(2, indi))


        x = torch.cat((seq,pos,struct),dim=2)
        x = self.transformer(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


from pytorch_metric_learning.miners import BaseMiner
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

class semisupertripletminer(BaseMiner):
    '''
    we try to build a tripplet loss minder that can handle -1 classes
    those are supposed to be far fom what we know
    '''
    def __init__(self, margin=0.1, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        mat = self.distance(embeddings, ref_emb)
        a,p,n = lmu.get_all_triplets_indices(labels, ref_labels)
        # ok here i should remove some things, where aaa is of class -1; i remove the entry.
        triplet_mask = labels[a] != -1
        a,p,n =  a[triplet_mask], p[triplet_mask], n[triplet_mask]
        pos_pairs = mat[a, p]
        neg_pairs = mat[a, n]
        triplet_margin = pos_pairs - neg_pairs if self.distance.is_inverted else neg_pairs - pos_pairs
        triplet_mask = triplet_margin <= self.margin
        return a[triplet_mask], p[triplet_mask], n[triplet_mask]



from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
distance = distances.CosineSimilarity()
reducer = reducers.ThresholdReducer(low=0)
loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
# mining_func = miners.TripletMarginMiner( margin=0.2, distance=distance, type_of_triplets="semihard")
mining_func = semisupertripletminer(distance=distance)
accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

def loss(pred,target):
    indices_tuple = mining_func(pred, target['trainlabel'])
    loss = loss_func(pred, target['testlabel'], indices_tuple)
    return loss

from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('module://matplotlib-backend-sixel')
import umap
from sklearn.decomposition import PCA
class METRIC(Metric):
    def __init__(self):
        self.reset()
    def reset(self):
        self.x,self.y = [],[]
    def accumulate(self, learn):
        x = learn.pred
        y = learn.y['testlabel']
        self.x.append(x)
        self.y.append(y)

    @property
    def value(self):

        x,y = torch.cat(self.x,0),torch.cat(self.y,0)
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        pca = PCA(n_components = 2).fit_transform(x)
        plt.scatter(*pca.T, c=y)
        plt.show()
        plt.close()

        # silhou = silhouette_score(test_embeddings,  test_labels)
        # print(f"{silhou= }")
        # ari = adjusted_rand_score( KMeans(n_clusters=len(np.unique(test_labels))).fit_predict(test_embeddings), test_labels)
        # print(f"{ ari=}")
        # return silhou, ari
        return 0 #silhou


import dirtyopts
docs = '''
--dev int 0    # cuda device id
--batchsize int 256  #batchsize
--dim int 128  #embedding dimension (64 are reserved for structure)
'''
args = dirtyopts.parse(docs)

fname = 'example0'
PATH = '../rnaformer_rfam.plk'
OUT = './'
bs = args.batchsize
num_workers = 2
SEED = 2023
nfolds = 4
device = f'cuda:{args.dev}'



seed_everything(SEED)
os.makedirs(OUT, exist_ok=True)
df = pd.read_pickle(PATH)
from fastai.learner import Learner

for fold in [0]:

    # DATA LOADING
    ds_train = RNA_Dataset(df, mode='train', fold=fold, nfolds=nfolds)
    dl_train = DeviceDataLoader(torch.utils.data.DataLoader(ds_train, batch_size= bs,shuffle = True, num_workers=num_workers, persistent_workers=True), device)

    ds_val = RNA_Dataset(df, mode='eval', fold=fold, nfolds=nfolds)
    dl_val= DeviceDataLoader(torch.utils.data.DataLoader(ds_val, batch_size=bs, num_workers=num_workers), device)

    gc.collect()
    data = DataLoaders(dl_train, dl_val)
    model = RNA_Model(dim=args.dim)
    model = model.to(device)
    # learn = Learner(data, model, loss_func=loss,cbs=[GradientClip(3.0)], metrics=[METRIC()]).to_fp16()
    learn = Learner(data, model, loss_func=loss,cbs=[], metrics=[METRIC()]).to_fp16()
    # learn = Learner(data, model, loss_func=loss,cbs=[], metrics=[]).to_fp16()
    learn.fit_one_cycle(320, lr_max=5e-4, wd=0.05, pct_start=0.02)

    # later ....
    # torch.save(learn.model.state_dict(),os.path.join(OUT,f'{fname}_{fold}.pth'))
    # gc.collect()


# vi /home/stefan/.myconda/miniconda3/envs/rnafenv/lib/python3.10/site-packages/fastai/learner.py
  # 8 class AvgSmoothLoss(Metric):
  # 7     "Smooth average of the losses (exponentially weighted with `beta`)"
  # 6     def __init__(self, beta=0.98): self.beta = beta
  # 5     def reset(self):               self.count,self.val = 0,tensor(0.)
  # 4     def accumulate(self, learn):
  # 3         self.count += 1
  # 2         # self.val = torch.lerp(to_detach(learn.loss.mean()), self.val, self.beta)
  # 1         start = to_detach(learn.loss.mean())
# 511         end = self.val
  # 1         weight = self.beta
  # 2         self.val = start + weight* (end - start)
