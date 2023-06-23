from lmz import Map,Zip,Filter,Grouper,Range,Transpose
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier as RF
import traceback
import numpy as np
from yoda import filein, simpleMl
import networkx as nx
from ubergauss import tools as ut
from yoda import ali2graph
import eden.graph as eg
from scipy import sparse
from sklearn.metrics import pairwise, adjusted_rand_score
import matplotlib
matplotlib.use('module://matplotlib-sixel')
from matplotlib import pyplot as plt
import umap

clans = '''CL00110	mir-19	mir-363
CL00111	SSU_rRNA_bacteria	SSU_rRNA_archaea	SSU_rRNA_eukarya	SSU_rRNA_microsporidia	SSU_trypano_mito
CL00071	SNORD88	snR76	snoR118
CL00070	snosnR60_Z15	SNORD77	Afu_263
CL00073	snoR30	SNORD100
CL00072	SNORD96	SNORD2
CL00075	SNORD105	snoU105B
CL00074	SNORD101	snoR60
CL00077	SNORND104	snR58
CL00076	SNORD110	snoR14
CL00079	snR68	snoR27
CL00078	SNORA73	snR30
CL00068	SNORD62	snoR41	snR41
CL00069	SNORD79	SNORD113	SNORD74	snoR44_J54	snosnR64	snoMe28S-Cm2645	SNORD64	SNORD112	SNORD109A
CL00066	SNORD60	snoR1	snosnR48	snoMe28S-G3255	sn2903
CL00067	SNORD61	snoU61	SNORD125
CL00064	SNORD58	SNORD99
CL00065	snoZ159	SNORD59	snosnR54
CL00062	SNORD49	snoZ112	snoU49
CL00063	SNORD52	SNORD53_SNORD92	snoZ157	snR78
CL00060	SNORD44	snoZ102_R77
CL00061	SNORD46	snR63
CL00093	mir-183	mir-182	mir-263	mir-228
CL00092	mir-137	mir-234
CL00091	bantam	mir-81
CL00090	mir-74	mir-73
CL00097	mir-BART1	mir-BART3
CL00096	mir-515	mir-290	mir-302	miR-430
CL00095	mir-279	mir-996
CL00094	mir-216	mir-283
CL00099	MIR171_1	MIR171_2
CL00098	MIR169_2	MIR169_5
CL00019	SCARNA4	SCARNA15
CL00018	SCARNA24	SCARNA3
CL00013	7SK	Arthropod_7SK
CL00012	SAM	SAM-IV	SAM-I-IV-variant
CL00011	GlmZ_SraJ	GlmY_tke1
CL00010	Hammerhead_3	Hammerhead_1	Hammerhead_HH9	Hammerhead_II	Hammerhead_HH10
CL00017	IRES_HCV	IRES_Pesti
CL00016	FinP	traJ_5
CL00015	CRISPR-DR5	CRISPR-DR7	CRISPR-DR63	CRISPR-DR64
CL00014	CRISPR-DR2	CRISPR-DR4	CRISPR-DR14	CRISPR-DR17	CRISPR-DR25	CRISPR-DR43	CRISPR-DR66
CL00080	snoR53	snoR53Y
CL00081	snoZ279_R105_R108	snoU13
CL00082	snoU89	snoU85
CL00083	mir-6	mir-11
CL00084	mir-3	mir-318
CL00085	mir-16	mir-15
CL00086	mir-28	mir-708
CL00087	mir-34	mir-449
CL00088	mir-36	mir-42	mir-35
CL00089	mir-190	mir-50
CL00118	L31-Coriobacteria	L31-Corynebacteriaceae	L31-Firmicutes	L31-Gammaproteobacteria	L31-Actinobacteria
CL00008	U54	snoU54
CL00009	U6	U6atac
CL00125	Glycine	glycine-GGAnGA
CL00124	TD-1	TD-2
CL00123	Purine	MFR	2dG-II
CL00122	NiCo	NiCo-AnGGAG
CL00121	S15	S15-Methanomicrobia	S15-Halobacteria	S15-Flavobacteria
CL00120	Twister-P5	twister-P3	twister-P1
CL00001	tRNA	tmRNA	alpha_tmRNA	beta_tmRNA	cyano_tmRNA	tRNA-Sec	mt-tmRNA
CL00002	RNaseP_nuc	RNaseP_bact_a	RNaseP_bact_b	RNase_MRP	RNaseP_arch	RNase_P	RNaseP-T
CL00003	Metazoa_SRP	Bacteria_small_SRP	Fungi_SRP	Dictyostelium_SRP	Bacteria_large_SRP	Plant_SRP	Protozoa_SRP	Archaea_SRP	Bacteroidales_small_SRP
CL00004	Telomerase-vert	Telomerase-cil	Sacc_telomerase	Telomerase_Asco
CL00005	U1	U1_yeast	U11	Gl_U1
CL00006	U2	U12
CL00007	U4	U4atac
CL00035	SNORA36	snR36	snR44	S_pombe_snR36
CL00034	SNORA50	SNORA54	SNORA35	SNORA76
CL00037	SNORA48	snR86
CL00036	SNORA44	SNORA58	snR161	snR9
CL00031	SNORA21	snR10	S_pombe_snR10
CL00030	SNORA20	SNORA29
CL00033	SNORA28	snopsi18S-841	snR80
CL00032	SNORA27	SNORA26	snR42	S_pombe_snR42
CL00039	SNORA56	snR8
CL00038	SNORA66	snoR98	SNORA52	SNORA18	snoR80	snR49	snR5	S_pombe_snR5
CL00131	Flavi_ISFV_repeat_Ra_Rb	Flavi_ISFV_repeat_Rb	Flavi_ISFV_repeat_Ra
CL00132	snoR9	snoPyro_CD	sR47	sR60
CL00127	c-di-GMP-II	c-di-GMP-II-GAG
CL00126	c-di-GMP-I	c-di-GMP-I-GGC	c-di-GMP-I-UAU
CL00022	SNORA3	snR11
CL00023	SNORA4	snR83
CL00020	SL1	SL2
CL00021	SNORA16	SNORA2	snopsi28S-3327	snR189	snR46	S_pombe_snR46
CL00026	SNORA8	snR31
CL00027	SNORA9	snR33	snR43	S_pombe_snR33
CL00024	SNORA5	snR85
CL00025	SNORA7	snR81
CL00028	SNORA13	snR35	S_pombe_snR35
CL00029	SNORA43	SNORA17
CL00101	Cobalamin	AdoCbl_riboswitch	AdoCbl-variant
CL00100	U3	Fungi_U3	Plant_U3	ACEA_U3
CL00103	SNORD11	SNORD11B
CL00102	group-II-D1D4-1	group-II-D1D4-2	group-II-D1D4-3	group-II-D1D4-4	group-II-D1D4-5	group-II-D1D4-6	group-II-D1D4-7
CL00105	SraC_RyeA	SdsR_RyeB
CL00104	HBV_epsilon	AHBV_epsilon
CL00106	CsrB	CsrC	McaS	PrrB_RsmZ	RsmY	TwoAYGGAY	rsmX	Ysr186_sR026_CsrC	RsmW
CL00108	suhB	ar15
CL00129	cHP	DENV_SLA	Flavivirus-5UTR
CL00128	C4	c4-2	c4-a1b1
CL00057	SNORD39	snoZ7	snoZ101	SNORD65	snoR77Y	snR77
CL00056	SNORD35	snoZ161_228	snR73
CL00055	SNORD34	snR62
CL00054	SNORD33	snoZ196	SNORD51	snosnR55	snoMe18S-Um1356	snoMe28S-Am982	snR39	snR40
CL00053	SNORD31	snoZ17	snR67	snoR35
CL00052	SNORD30	snoU30
CL00051	SNORD36	SNORD29	snoZ223	SNORD38	snosnR69	snosnR61	snosnR71	SNORD78	snoR69Y	snR47	snoU36a
CL00050	SNORD26	SNORD81
CL00116	aCoV-5UTR	bCoV-5UTR	gCoV-5UTR	dCoV-5UTR	Sarbecovirus-5UTR
CL00117	s2m	Corona_pk3	aCoV-3UTR	bCoV-3UTR	gCoV-3UTR	dCoV-3UTR	Sarbecovirus-3UTR
CL00114	LhrC	rli22	rli33
CL00115	DUF805b	DUF805
CL00112	5_8S_rRNA	LSU_rRNA_archaea	LSU_rRNA_bacteria	LSU_rRNA_eukarya	LSU_trypano_mito
CL00113	5S_rRNA	mtPerm-5S
CL00059	SNORD43	snR70
CL00058	SNORD57	SNORD41	snR51
CL00044	SNORD12	snR190
CL00045	SNORD15	snR75	snR13	snoZ5
CL00046	SNORD16	snR87
CL00047	SNORD18	snoU18
CL00040	SNORA62	snR3	snR82	S_pombe_snR3
CL00041	SNORA64	snR37
CL00042	SNORA65	snR34	snoR2
CL00043	SNORA74	snR191
CL00119	S4-Fusobacteriales	S4-Bacteroidia	S4-Clostridia	S4-Flavobacteria
CL00048	SNORD19	SNORD19B
CL00049	SNORD25	snR56f'''

def getlabels(ali):
    y = np.zeros(len(ali))
    for i,e in enumerate(clans.split(f'\n')):
        rnaid = e.split()[1:]
        for j,a in enumerate( ali):
            if a.gf[f'ID'][3:] in rnaid:
               y[j] = i + 1
    return y



def vectorize(alignments,min_rd = 1, ignorevectorlabels = False, mp = False):
    vectorizer = lambda x: eg.vectorize([x.graph], discrete = ignorevectorlabels, min_r = min_rd, min_d = min_rd) # normalization=False, inner_normalization=False)
    mapper = ut.xmap if mp else Map
    vectors = mapper(vectorizer, alignments)
    vectors = sparse.vstack(vectors)
    return vectors


def get_ali_dist():
    # iconv -f ISO-8859-1 -t UTF-8 Rfam.seed > Rfam.seed.utf8
    alignments = filein.readseedfile(f'/home/ubuntu/Rfam.seed.utf8')
    graphs = ut.xmap(ali2graph.rfamseedfilebasic, alignments)
    vectors = vectorize(graphs)
    dist = pairwise.linear_kernel(vectors)
    return alignments, dist , vectors

########################
# so, first we cluster all the right alignments
##########################

def prepdata(full = False):
    alignments = filein.readseedfile(ut.fixpath( f'~/Rfam.seed.utf8'))
    labels = getlabels(alignments)

    if not full:
        oklabel = labels != 0
        labels = labels[oklabel]
        alignments = [a for a,ok in zip(alignments,labels) if ok]

    alignments = filein.addcov_rfam(alignments)
    alignments = ut.xmap(ali2graph.rfam_clean, alignments)

    return labels, alignments


def embed(distances, n_dim = 2):
    return umap.UMAP(n_components=n_dim,metric='precomputed',n_neighbors=2).fit_transform(distances)
def scatter(X, y):
    plt.scatter(*X.T, c = y)
    plt.show()


from sklearn.metrics import silhouette_score

from structout.rnagraph import RNAprint

def showgraph(ali):
    graph = nx.relabel_nodes(ali.graph,mapping = dict(zip(ali.graphnodes, Range(ali.graphnodes))),copy = True)
    RNAprint(graph, size = 2)
    return ali.graph
def shownodes(ali):
    for n in ali.graph.nodes:
        print(ali.graph.nodes[n])

from yoda import nearneigh as  nn
def plotneighbors(alignments,dist, alignment_id = 5, num = 3):


        def plotalign(a1,a2):

            # _fancyalignment(seq1,seq2,str1,str2)
            nn._fancyalignment(a1.graph.graph['sequence'], a2.graph.graph['sequence'], a1.graph.graph['structure'], a2.graph.graph['structure'])


        for i in np.argsort(dist[alignment_id])[1:num+1]:
            plotalign(alignments[alignment_id], alignments[i])


def get_distance_method(distance_measure):
    if distance_measure == 'euclidean':
        di = pairwise.euclidean_distances
    elif distance_measure == 'overlap':
        di = simpleMl.overlap
    elif distance_measure == 'overlap_nonorm':
        di = lambda x: 1 - simpleMl.overlap(x,nonorm=True)
    elif distance_measure == 'linkern':
        di = lambda x: 10 - pairwise.linear_kernel(x)
    else:
        assert False, 'distance_measure invalid'
    return di


import grakel


def convert_to_grakel_graph(graph: nx.DiGraph) -> grakel.Graph:
    dol = nx.convert.to_dict_of_lists(graph)
    node_attr = dict()
    for nid, attr in graph.nodes(data=True):
        node_attr[nid] = np.asarray([attr['label']]+attr['vec'].tolist())

    result = grakel.Graph(dol, node_labels=node_attr, graph_format='dictionary')
    for node in [n for n in dol.keys() if n not in result.edge_dictionary.keys()]:
        result.edge_dictionary[node] = {}
    return result

def grakel_vectorize(alignment, vectorizer= 'WeisfeilerLehman'):
    g =  convert_to_grakel_graph( alignment.graph)
    d = {'normalize' : False, 'sparse' : True}
    return eval('grakel'.vectorizer)(**d).fit_transform(g)

# def grakel_vectorize(alignment, vectorizer= None):
#     g =  convert_to_grakel_graph( alignment.graph)
#     if vectorizer is None:
#         vectorizer = grakel.WeisfeilerLehman
#     for ker in [kernellists... ]..
#     return eval('grakel'.vectori)(n_iter= 4, normalize = False, sparse = True).fit_transform(g)




def run(labels, alignments,
        RY_thresh = .01, conservation = [.50,.75,.90,.95],
        covariance = .05, sloppy = False, fake_nodes = False, vectorizer = 'WeisfeilerLehman',
        distance_measure = 'euclidean',min_rd = 2,n_dim = 6, mp = False):


    mapper = ut.xmap if mp else Map
    alignments = mapper( lambda ali:ali2graph.rfam_graph_decoration(ali, RY_thresh = .1,
                          conservation = conservation,
                          covariance = covariance,
                          sloppy = sloppy,
                          fake_nodes = fake_nodes),
                          alignments)

    # showgraph(alignments[4])
    # breakpoint()
    # vectors = vectorize_debug(alignments)

    # vectors = vectorize(alignments,
    #                     min_rd = min_rd,
    #                     ignorevectorlabels= False,
    #                     mp=mp)

    vectors  = mapper(lambda vectorizer:grakel_vectorize(x,vectorizer),alignments)

    di = get_distance_method(distance_measure)
    dist = di(vectors)
    X = embed(dist, n_dim= n_dim)

    # plotneighbors(alignments,dist, alignment_id = 5, num  = 4)
    # scatter(X, labels)
    # print(f"{ simpleMl.permutation_score(X, labels) = }")

    return silhouette_score(X, labels)
    return silhouette_score(X, labels), adjusted_rand_score(
                        KMeans(n_clusters=len(np.unique(labels))).fit_predict(X),
                        labels)


############
# get goodd features
#####################
def subset(l,a, mask):
    # return [aa for ll,aa in zip(mask,a) if  ll], l[mask]
    return a[mask], l[mask]
def split(l,a,testlab):
    yes = [ll in testlab for ll in l]
    return subset(l,a,[False if a else True for a in yes]),subset(l,a,yes)



def get_features(X,y,n_ft = 100):
    # clf = RF(n_estimators = 100,max_features=None).fit(X,y)
    clf = RF(n_estimators = 200).fit(X,y)
    featscores =  clf.feature_importances_
    # print(f"{ featscores=}")
    return ut.binarize(featscores, n_ft)

def eval_ft(ft,X,y):
    # X=X.todense()
    X = X[:,ft==1]
    X = pairwise.euclidean_distances(X)
    X = embed(X, n_dim= 6)
    return silhouette_score(X, y)



def decorate(ali):
    return  ut.xmap( lambda ali:ali2graph.rfam_graph_decoration(ali, RY_thresh = .2,
                          covariance = 0.05,
                          sloppy = False,
                          fake_nodes = False),
                          ali)


def supervised(l,a, n_ft = 100):

    list_of_classes = np.unique(l)
    np.random.shuffle(list_of_classes)

    # a = ut.xmap(ali2graph.rfam_graph_structure_deco, a)
    a = decorate(a)
    X = vectorize(a,min_rd=1, mp= True)
    test_scores = []
    for testlabels in np.split(list_of_classes,3):
        train, test = split(l,X, testlabels)
        ft = get_features(*train, n_ft)
        test_scores.append( (eval_ft(ft, *test)) )
        print(f"{ eval_ft(ft, *train) = }")
    print(f"{ np.mean(test_scores)= }")






############
#   grip optimization
##########
from ubergauss import optimization as opti
grid = {
        #'RY_thresh' : np.linspace(0.1, 0.3, 3),
        'covariance': [.05, .1],
        # 'distance_measure': 'euclidean '.split(),
        'fake_nodes': [False,True],
        # 'sloppy': [False,True],
        'mp': [False],
        'min_rd': [1,2]}

def optimize():
    labels, alignments = prepdata(full = False)
    df =  opti.gridsearch(run, grid, (labels, alignments))

    print(df.corr(method='pearson'))
    # print(df.corr(method='spearman'))
    # print(df.corr(method='kendall'))

    print(df.sort_values(by = 'score'))

def run_st(labels, alignments, distance_measure = 'euclidean',min_rd = 1,n_dim = 6, mp = True):
    mapper = ut.xmap if mp else Map
    alignments = mapper(ali2graph.rfam_graph_structure_deco, alignments)

    # showgraph(alignments[4])
    # breakpoint()
    # vectors = vectorize_debug(alignments)

    vectors = vectorize(alignments, min_rd = min_rd,ignorevectorlabels= False,mp=mp)

    # z= alignments[0].graph
    # for n in z:
    #     print(f"{z.nodes[n]['vec'] = }")

    # vectors = vectorize(alignments, min_rd = min_rd,ignorevectorlabels= True,mp=mp)
    # print(vectors[0].data.shape)

    di = get_distance_method(distance_measure)
    dist = di(vectors)
    X = embed(dist, n_dim= n_dim)
    # plotneighbors(alignments,dist, alignment_id = 5, num  = 4)
    # scatter(X, labels)
    # print(f"{ simpleMl.permutation_score(X, labels) = }")

    return silhouette_score(X, labels)
