from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
import ubergauss.tools as ut

import networkx.linalg as nxlin
import numpy as np





import torch,math
def make_sincos_index(max_len, dim=4):
    # indices = np.repeat ( Range(maxlen), maxlen).reshape((maxlen,maxlen))
    pe = torch.zeros((max_len, dim))
    # print(pe)
    position = torch.arange(0, max_len).unsqueeze(1).type(torch.FloatTensor)
    div_term = torch.exp( torch.arange(0, dim, 2).type(torch.FloatTensor) * -(math.log(10000.0) / dim))

    # print(torch.sin(position * div_term))
    pe[:,0::2] = torch.sin(position * div_term)
    pe[:,1::2] = torch.cos(position * div_term)

    pattern =  pe.numpy()
    return  np.repeat(pattern, max_len).reshape((4,max_len,max_len))


def test_make_sincos_index():
    layers = make_sincos_index(20)
    import structout as so
    # print (layers.shape)
    for layer in layers:
        so.heatmap(layer)

def makematrix(graph, maxlen = None):

    startpadding = 5
    endpadding = 5

    # add padding nodes
    missing_nodes = maxlen - len(graph)
    missing_nodes += endpadding # extra padding
    maxlen += startpadding+endpadding

    for i in range(missing_nodes):
        graph.add_node(max(graph.nodes)+1 , label = 'N')

    for i in range(startpadding):
        graph.add_node( -i-1 , label = 'N')

    assert len(graph)== maxlen, f"{len(graph)=}, {maxlen=}"
    sorted_nodes = sorted(graph.nodes)
    adjacency_matrix = nxlin.adjacency_matrix(graph,nodelist=sorted_nodes).todense()
    # now we need to add another axis and add node infos
    res = np.zeros((17, maxlen, maxlen),dtype = np.float32)
    indices = make_sincos_index(maxlen)

    def makeonehot(char):
        return np.repeat ( [ 1 if graph.nodes[n]['label'] == char else 0 for n in sorted_nodes ] , maxlen).reshape((maxlen,maxlen))

    ONEHOT = np.array( Map(makeonehot, 'AUGC'))
    res[:4] = ONEHOT
    res[4:8] = np.transpose(ONEHOT, (0,2,1))

    res[8:12] = indices
    res[12:16] = np.transpose(indices, (0,2,1))
    res[16] = adjacency_matrix

    return res


def toarray(alis):
    graphs = [a.graph for a in alis]
    maxlen = max(map(len, graphs))
    return ut.xmap(makematrix, graphs, maxlen = maxlen)


def len_hist(alis):
    graphlens = [len(a.graph) for a in alis]
    import matplotlib
    matplotlib.use('module://matplotlib-backend-sixel')
    import seaborn as sns
    sns.histplot(graphlens)
    matplotlib.pyplot.show()
    matplotlib.pyplot.close()

def test_toarray():
    from yoda import alignments
    a,l = alignments.load_rfam(add_cov='')
    a,l = alignments.size_filter(a,l,400)

    print(f"{len(a)=}")
    atensor = toarray(a[:2])[0]
    import structout as so
    for layer in atensor:
        so.heatmap(layer)
    len_hist(a)

if __name__ == f"__main__":
    test_toarray()




import ubergauss.optimization as uo
import torch
from torch.utils.data import Dataset, DataLoader






##################
# old loader that builds all matrices instantly
####################
class CustomDataset_old(Dataset):
    def __init__(self, numpy_array_list, labels):
        self.data = numpy_array_list
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return torch.from_numpy(sample), torch.tensor(label)

def makeloader_old(arrays_labels, batch_size):
    return   DataLoader(CustomDataset(*arrays_labels), batch_size=batch_size, shuffle=True)

def torchloader_old(batch_size, alignments, labels):
    X  = np.array(toarray(alignments))

    train, test = next(uo.groupedCV(n_splits = 3).split(_,labels,labels))
    labels = np.array(labels)
    tr = X[train], labels[train]
    te = X[test], labels[test]
    return makeloader(tr, batch_size), CustomDataset(*tr), CustomDataset(*te)










##################
# improved loader that builds matrixces on the fly
####################
class CustomDataset(Dataset):
    def __init__(self, alignments, labels, maxlen):
        self.maxlen= maxlen
        self.data = alignments
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = makematrix(self.data[idx], self.maxlen)
        label = self.labels[idx]
        return torch.from_numpy(sample), torch.tensor(label)

def makeloader(ali,label, batch_size, maxlen):
    return   DataLoader(CustomDataset(ali,label, maxlen), prefetch_factor = 1,
                        batch_size=batch_size, shuffle=True, num_workers=1)

def torchloader(batch_size, graphs, labels):

    maxlen = max(map(len, graphs))
    train, test = next(uo.groupedCV(n_splits = 3).split(graphs,labels,labels))
    labels = np.array(labels)
    tr = labels[train]
    te = labels[test]
    train_alignments  = [graphs[i] for i in train]
    return makeloader(train_alignments,tr, batch_size, maxlen), CustomDataset(train_alignments, tr, maxlen), CustomDataset([graphs[i] for i in test], te, maxlen)












