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

def dict_to(x, device='cuda'):
    return {k:x[k].to(device) for k in x}

def to_device(x, device='cuda'):
    return tuple(dict_to(e,device) for e in x)

class DeviceDataLoader:
    def __init__(self, dataloader, device='cuda'):
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
        self.seq_map = {'A':0,'C':1,'G':2,'U':3}
        self.Lmax = 400

        labels = df['set'].values
        split = list(groupedCV(n_splits=nfolds, randseed=seed).split(df,labels, groups=labels))[fold][0 if mode=='train' else 1]

        df = df.iloc[split].sample(frac=1).reset_index(drop=True)
        # df = df.iloc[split].reset_index(drop=True)

        # our df has 'set' (labels), sequence, pos1id, pos2id
        self.seq = df['sequence'].values
        self.labels = df['set'].values
        self.p1 = df['pos1id'].values
        self.p2 = df['pos2id'].values

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        seq = self.seq[idx]
        seq = [self.seq_map[s] for s in seq]
        seq = np.array(seq)
        seq = np.pad(seq, (0,self.Lmax-len(seq)))
        label = np.array(self.labels[idx])
        p1 =  np.array(self.p1[idx])
        p2 =  np.array(self.p2[idx])
        p1 = np.hstack((p1,[-1]*(self.Lmax-len(p1))), dtype = np.int32, casting = 'unsafe')
        p2 = np.hstack((p2,[-1]*(self.Lmax-len(p2))), dtype = np.int32, casting = 'unsafe')

        return {'seq':torch.from_numpy(seq), 'label':label, 'p1':p1, 'p2':p2}, {'label':label}



class RNA_Model(nn.Module):
    def __init__(self, dim=128, depth=12, head_size=32, **kwargs):
        super().__init__()
        self.emb = nn.Embedding(4,64)
        self.pos_enc = SinusoidalPosEmb(32)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=dim//head_size, dim_feedforward=4*dim,
                dropout=0.1, activation=nn.GELU(), batch_first=True, norm_first=True), depth)
        self.proj_out = nn.Linear(dim,2)


    def forward(self, x0):
        # note i removed the masking maybe that was too much cleaning
        x = x0['seq']
        p1,p2 = x0['p1'], x0['p2']


        pos = torch.arange(x.shape[1], device=x.device)
        pos= pos.unsqueeze(0)
        pos = self.pos_enc(pos) # bsx400
        pos = torch.repeat_interleave(pos, 128, dim=0 )





        struct = torch.zeros((128,400,32), device = pos.device)
        # for i,(pp1,pp2) in enumerate(zip(p1,p2)):
        #     # pp1 = pp1[pp1!=-1]
        #     # pp2 = pp2[pp2!=-1]
        #     pp1 = pp1!=-1
        #     pp2 = pp2!=-1
        #     struct[i,pp1] = pos[i,pp2]
        #     struct[i,pp2] = pos[i,pp1 ]
        struct[p1!=-1] = pos[p2!=-1]
        struct[p2!=-1] = pos[p1!=-1]

        x = self.emb(x) # bs 400 dim
        x = torch.cat((x,pos,struct),2)
        x = self.transformer(x)
        x = self.proj_out(x)

        return x


def loss(pred,target):
    p = pred[target['mask'][:,:pred.shape[1]]]
    y = target['react'][target['mask']].clip(0,1)
    loss = F.l1_loss(p, y, reduction='none')
    loss = loss[~torch.isnan(loss)].mean()

    return loss

class MAE(Metric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.x,self.y = [],[]

    def accumulate(self, learn):
        x = learn.pred[learn.y['mask'][:,:learn.pred.shape[1]]]
        y = learn.y['react'][learn.y['mask']].clip(0,1)
        self.x.append(x)
        self.y.append(y)

    @property
    def value(self):
        x,y = torch.cat(self.x,0),torch.cat(self.y,0)
        loss = F.l1_loss(x, y, reduction='none')
        loss = loss[~torch.isnan(loss)].mean()
        return loss


fname = 'example0'
PATH = '../rnaformer_rfam.plk'
OUT = './'
bs = 128
num_workers = 2
SEED = 2023
nfolds = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'



seed_everything(SEED)
os.makedirs(OUT, exist_ok=True)
df = pd.read_pickle(PATH)
from fastai.learner import Learner

for fold in [0]: # running multiple folds at kaggle may cause OOM

    # DATA LOADING
    ds_train = RNA_Dataset(df, mode='train', fold=fold, nfolds=nfolds)
    dl_train = DeviceDataLoader(torch.utils.data.DataLoader(ds_train, batch_size= bs, num_workers=num_workers, persistent_workers=True), device)

    ds_val = RNA_Dataset(df, mode='eval', fold=fold, nfolds=nfolds)
    dl_val= DeviceDataLoader(torch.utils.data.DataLoader(ds_val, batch_size=bs, num_workers=num_workers), device)

    gc.collect()
    data = DataLoaders(dl_train, dl_val)


    model = RNA_Model()
    model = model.to(device)
    learn = Learner(data, model, loss_func=loss,cbs=[GradientClip(3.0)], metrics=[MAE()]).to_fp16()
    learn.fit_one_cycle(32, lr_max=5e-4, wd=0.05, pct_start=0.02)

    # later ....
    # torch.save(learn.model.state_dict(),os.path.join(OUT,f'{fname}_{fold}.pth'))
    # gc.collect()






