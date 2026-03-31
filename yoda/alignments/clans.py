import numpy as np

'''
i need functions to
load rf15
load rf15 clans without rf14
stats
'''


def read_clans(clans, fixrf14=False):
    '''
    produces a dictionary {rna_name -> clusterlabel}
    '''
    d = {}
    d_reverse ={} # needed for rf14 fix
    for i,e in enumerate(clans.split(f'\n')):
        line = e.split()
        d[line[0]] = line[1:]
        # i just add the reverse lookup to make rf14 easier
        for fam in line[1:]:
            d_reverse[fam] = line[0]

    if fixrf14:
        # since we use the clan list from rf14 but the rfam15 seed file, we need to apply some corrections:
        old_names = ['SAM-I-IV-variant', 'MFR', 'AdoCbl_riboswitch','AdoCbl-variant']
        new_names = ['SAM-I-IV',         '2dG-I', 'AdoCbl','AdoCbl-II']
        for old,new in zip(old_names,new_names):
            clan = d_reverse.pop(old, None)
            d[clan].remove(old)
            d[clan].append(new)

    return d








def fam_to_id(d):
    r = {}
    for i,(k,v) in enumerate(d.items()):
        for fam in v:
            r[fam] = i+1
    return r

def assign_labels(alignments, label_dict):
    label_dict =  fam_to_id(label_dict)
    y = np.zeros(len(alignments),dtype=int)
    for j,a in enumerate( alignments):
        ali_label = a.gf[f'ID'][3:]
        label = label_dict.get( ali_label ,  0 )
        # if 'snR56' in ali_label: print(f"{ali_label=}")
        y[j] = label
        label_dict.pop(ali_label, None)
    if label_dict:
        print(f"i expected these to be in the seedfile, as they are in the class list: {label_dict=}")
    return y



def getlabels_rfam15(alignments):
    '''
    -> labels for all ze alignments ,
    '''
    label_dict, burned = merge_clans(mode = 'test')
    labels = assign_labels(alignments,label_dict)
    burn = [ a.gf[f'ID'][3:] in burned for a in alignments]
    burn = {i for i, val in enumerate(burn) if val}
    return labels, burn

def getlabels(alignments):
    '''
    matches the ID field in the alignments with the "clans"-cluster-list
    returns cluster-labels for rf15 but only clans in rf14
    '''
    label_dict = merge_clans(mode = 'train')
    return assign_labels(alignments,label_dict)



def merge_clans(mode, print_extenders= False):
    rf15 = read_clans(clans_rfam15, fixrf14=False)
    rf14 = read_clans(clans, fixrf14=True)
    'basically 14 determines whats in the train and test set'
    extender = []
    justnew = []
    if mode == 'train':
        # return cl from 14
        return rf14

    elif mode == 'test':
        # rf15 but we remove stuff that is already in 14...
        burned = []
        # every new clan
        for k in list(rf15.keys()):
            # if its new, its ok ...
            if k not in rf14:
                justnew.append(k)
                continue
            # else we check every entry
            for v in list(rf15[k]):
                if v in rf14[k]:
                    burned.append(v)
                else: # we keep it
                    extender.append((k,v))
        if print_extenders:
            print(f"{extender=}")
            print(f"{justnew=}")
        # print(len(extender))
        return rf15, set(burned)
    else:
        assert False



def stats():

    def show(d):
        # print(f"{len(d)=} {sum([len(v) for v in d.values()])=}")
        print(f"Clans: {len(d)}  Families:{sum([len(v) for v in d.values()])}")

    rf15 = read_clans(clans_rfam15, fixrf14=False)
    print(f"rf15 total")
    show(rf15)

    print(f"rf14")
    d = merge_clans(mode='train')
    show(d)

    d, burn = merge_clans(mode='test', print_extenders= True)
    # print(f"test")
    # show(d)
    print(f"{len(burn)=}")

import matplotlib.pyplot as plt
def get_clan_avg_length(alignments):
    """
    Returns a dictionary {clan_id -> average_alignment_length}
    using Rfam 15.1 and long clans.

    # i need a function.
    #     takes alignments as input.
    #     loads labels 15.1 and the long ones -> readclans(rfam15+long)
    #     alignments have a.alignment.shape[1]  we need the average alignment length per  clan


    """

    assert len(alignments) > 1000, 'making sure you call this with all alignments... if you load the short list, the long ones are invisible'

    combined_clans = clans_rfam15 + '\n' + long_clans
    clan_dict = read_clans(combined_clans)
    # Map family name to clan ID
    fam_to_clan = {}
    for clan_id, families in clan_dict.items():
        for fam in families:
            fam_to_clan[fam] = clan_id
    clan_lengths = {}
    for a in alignments:
        fam_id = a.gf['ID'][3:]# if a.gf['ID'].startswith('RF') else a.gf['ID']
        clan_id = fam_to_clan.get(fam_id)

        if clan_id:
            if clan_id not in clan_lengths:
                clan_lengths[clan_id] = []
            clan_lengths[clan_id].append(a.alignment.shape[1])
    ret=  {k: np.mean(v) for k, v in clan_lengths.items()}
    #return ret
    plt.hist(ret.values(), bins=50)
    plt.show()
    breakpoint()
    #print(f"{np.array(ret.values()).mean() = }")
    # mean 198.00699664573838
    # std 347.42055167848196
    # CL00112	5_8S_rRNA	LSU_rRNA_archaea	LSU_rRNA_bacteria	LSU_rRNA_eukarya	LSU_trypano_mito -> 3730
    # CL00111	SSU_rRNA_bacteria	SSU_rRNA_archaea	SSU_rRNA_eukarya	SSU_rRNA_microsporidia	SSU_trypano_mito -> 1920
    # CL00004	Telomerase-vert	Telomerase-cil	Sacc_telomerase	Telomerase_Asco -> 1131

import pandas as pd
def rna_type_table():

    def show(d):
        # print(f"{len(d)=} {sum([len(v) for v in d.values()])=}")
        ismirna = lambda s: any(map( lambda z:s.startswith(z),'mir MIR let bantam'.split()))
        countmirna = lambda x: len([y for y in x if ismirna(y)])
        nummicro =  sum([countmirna(v) for v in d.values()])
        print(f"Clans: {len(d)}  Families:{sum([len(v) for v in d.values()])} Micro: {nummicro}")
        return len(d), sum([len(v) for v in d.values()]), nummicro

    #  collect data for old stuff
    print(f"rf14")
    d = merge_clans(mode='train')
    # breakpoint()
    data_old = show(d)



    d, burn = merge_clans(mode='test', print_extenders= True)
    # d is now rf15 complete.
    newcl_rf15 = ['CL00145', 'CL00144', 'CL00147', 'CL00146', 'CL00141', 'CL00140',
             'CL00143', 'CL00142', 'CL00149', 'CL00148', 'CL00150', 'CL00135',
             'CL00136', 'CL00137', 'CL00133', 'CL00138', 'CL00139']
    # this is 15.1
    newcl = ['CL00133', 'CL00135', 'CL00136', 'CL00137', 'CL00138', 'CL00139', 'CL00140', 'CL00141', 'CL00142', 'CL00143', 'CL00144', 'CL00145', 'CL00146', 'CL00147', 'CL00148', 'CL00149', 'CL00150', 'CL00151', 'CL00152', 'CL00153', 'CL00154', 'CL00155']


    data_new = show({a:b for a,b in d.items() if a in newcl})
    print(f"{len(burn)=}")
    # breakpoint()
    data_extender_rf15  = 3, 8, 7 # the stats() function will help you calculatethis
    data_extender  = 4, 9, 7
    # extenders 15.1
    # [('CL00088', 'mir-35_2'),
    #  ('CL00088', 'mir-36_2'),
    #  ('CL00088', 'mir-39'),
    #  ('CL00098', 'MIR169_3'),
    #  ('CL00098', 'MIR169_6'),
    #  ('CL00098', 'MIR169_7'),
    #  ('CL00098', 'MIR169_8'),
    #  ('CL00106', 'RsmV'),
    #  ('CL00127', 'c-di-GMP-II-GCG')]


    headers = 'Old New Extenders'.split()

    #make a pandas table and print it. the data is data_old data_new data_extender, column index is headers, row index is the data name
    df = pd.DataFrame(
        [data_old, data_new, data_extender],
        index=headers,
        columns=['Clans', 'Families', 'Micro']
    ).T
    print(df.to_latex())

# the first 3 clans are very long rnas...
# CL00112	5_8S_rRNA	LSU_rRNA_archaea	LSU_rRNA_bacteria	LSU_rRNA_eukarya	LSU_trypano_mito
# CL00111	SSU_rRNA_bacteria	SSU_rRNA_archaea	SSU_rRNA_eukarya	SSU_rRNA_microsporidia	SSU_trypano_mito
# CL00004	Telomerase-vert	Telomerase-cil	Sacc_telomerase	Telomerase_Asco

clans = '''CL00110	mir-19	mir-363
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
CL00049	SNORD25	snR56'''






## we include the too long clans again here...
## some clans have changed:
# changed_clan: ['MIR169_2', 'MIR169_5', 'MIR169_3', 'MIR169_6', 'MIR169_7', 'MIR169_8']
# changed_clan: ['SAM', 'SAM-IV', 'SAM-I-IV']
# changed_clan: ['mir-36', 'mir-42', 'mir-35', 'mir-35_2', 'mir-36_2', 'mir-39']
# changed_clan: ['Purine', '2dG-I', '2dG-II']
# changed_clan: ['c-di-GMP-II', 'c-di-GMP-II-GAG', 'c-di-GMP-II-GCG']
# changed_clan: ['Cobalamin', 'AdoCbl', 'AdoCbl-II']


# CL00111	SSU_rRNA_bacteria	SSU_rRNA_archaea	SSU_rRNA_eukarya	SSU_rRNA_microsporidia	SSU_trypano_mito
# CL00112	5_8S_rRNA	LSU_rRNA_archaea	LSU_rRNA_bacteria	LSU_rRNA_eukarya	LSU_trypano_mito
# CL00004	Telomerase-vert	Telomerase-cil	Sacc_telomerase	Telomerase_Asco


# let bantam are the only mirs that are not caled mir. i handchecked: https://rfam.org/search?q=entry_type:%22Family%22%20AND%20rna_type:%22miRNA%22%20AND%20NOT%20%22mir%22

clans_rfam15_0 = '''CL00110	mir-19	mir-363
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
CL00145	MIR1520	MIR4372
CL00144	MIR4387	MIR4371
CL00147	mir-574	mir-9201
CL00146	mir-8799	mir-8791
CL00141	mir-H7	mir-H20
CL00140	mir-511	mir-506
CL00143	MIR827	MIR827_2
CL00142	mir-64	mir-2851	mir-2733
CL00149	mir-154	mir-1197	mir-368	mir-379	mir-889	mir-3578	mir-329	mir-485
CL00148	let-7	mir-3596
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
CL00098	MIR169_2	MIR169_5	MIR169_3	MIR169_6	MIR169_7	MIR169_8
CL00150	MIR162_1	MIR162_2
CL00019	SCARNA4	SCARNA15
CL00018	SCARNA24	SCARNA3
CL00013	7SK	Arthropod_7SK
CL00012	SAM	SAM-IV	SAM-I-IV
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
CL00088	mir-36	mir-42	mir-35	mir-35_2	mir-36_2	mir-39
CL00089	mir-190	mir-50
CL00118	L31-Coriobacteria	L31-Corynebacteriaceae	L31-Firmicutes	L31-Gammaproteobacteria	L31-Actinobacteria
CL00008	U54	snoU54
CL00009	U6	U6atac
CL00125	Glycine	glycine-GGAnGA
CL00124	TD-1	TD-2
CL00123	Purine	2dG-I	2dG-II
CL00122	NiCo	NiCo-AnGGAG
CL00121	S15	S15-Methanomicrobia	S15-Halobacteria	S15-Flavobacteria
CL00120	Twister-P5	twister-P3	twister-P1
CL00001	tRNA	tmRNA	alpha_tmRNA	beta_tmRNA	cyano_tmRNA	tRNA-Sec	mt-tmRNA
CL00002	RNaseP_nuc	RNaseP_bact_a	RNaseP_bact_b	RNase_MRP	RNaseP_arch	RNase_P	RNaseP-T
CL00003	Metazoa_SRP	Bacteria_small_SRP	Fungi_SRP	Dictyostelium_SRP	Bacteria_large_SRP	Plant_SRP	Protozoa_SRP	Archaea_SRP	Bacteroidales_small_SRP
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
CL00135	mir-251	mir-252
CL00136	mir-465	mir-509	mir-743	mir-8908	mir-890	mir-507	mir-513
CL00137	mir-92	mir-310
CL00131	Flavi_ISFV_repeat_Ra_Rb	Flavi_ISFV_repeat_Rb	Flavi_ISFV_repeat_Ra
CL00132	snoR9	snoPyro_CD	sR47	sR60
CL00133	mir-31	mir-72
CL00138	mir-8	mir-236
CL00139	MIR2863	MIR5070
CL00127	c-di-GMP-II	c-di-GMP-II-GAG	c-di-GMP-II-GCG
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
CL00101	Cobalamin	AdoCbl	AdoCbl-II
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
CL00049	SNORD25	snR56'''



# the long clans:
long_clans = ''' CL00111	SSU_rRNA_archaea	SSU_rRNA_bacteria	SSU_rRNA_eukarya	SSU_rRNA_microsporidia	SSU_trypano_mito
CL00112	5_8S_rRNA	LSU_rRNA_archaea	LSU_rRNA_bacteria	LSU_rRNA_eukarya	LSU_trypano_mito
CL00004	Sacc_telomerase	Telomerase-cil	Telomerase-vert	Telomerase_Asco'''

# this is 15.1
clans_rfam15 = '''CL00001	alpha_tmRNA	beta_tmRNA	cyano_tmRNA	mt-tmRNA	tmRNA	tRNA	tRNA-Sec
CL00002	RNaseP-T	RNaseP_arch	RNaseP_bact_a	RNaseP_bact_b	RNaseP_nuc	RNase_MRP	RNase_P
CL00003	Archaea_SRP	Bacteria_large_SRP	Bacteria_small_SRP	Bacteroidales_small_SRP	Dictyostelium_SRP	Fungi_SRP	Metazoa_SRP	Plant_SRP	Protozoa_SRP
CL00005	Gl_U1	U1	U11	U1_yeast
CL00006	U12	U2
CL00007	U4	U4atac
CL00008	snoU54	U54
CL00009	U6	U6atac
CL00010	Hammerhead_1	Hammerhead_3	Hammerhead_HH10	Hammerhead_HH9	Hammerhead_II
CL00011	GlmY_tke1	GlmZ_SraJ
CL00012	SAM	SAM-I-IV	SAM-IV
CL00013	7SK	Arthropod_7SK
CL00014	CRISPR-DR14	CRISPR-DR17	CRISPR-DR2	CRISPR-DR25	CRISPR-DR4	CRISPR-DR43	CRISPR-DR66
CL00015	CRISPR-DR5	CRISPR-DR63	CRISPR-DR64	CRISPR-DR7
CL00016	FinP	traJ_5
CL00017	IRES_HCV	IRES_Pesti
CL00018	SCARNA24	SCARNA3
CL00019	SCARNA15	SCARNA4
CL00020	SL1	SL2
CL00021	snopsi28S-3327	SNORA16	SNORA2	snR189	snR46	S_pombe_snR46
CL00022	SNORA3	snR11
CL00023	SNORA4	snR83
CL00024	SNORA5	snR85
CL00025	SNORA7	snR81
CL00026	SNORA8	snR31
CL00027	SNORA9	snR33	snR43	S_pombe_snR33
CL00028	SNORA13	snR35	S_pombe_snR35
CL00029	SNORA17	SNORA43
CL00030	SNORA20	SNORA29
CL00031	SNORA21	snR10	S_pombe_snR10
CL00032	SNORA26	SNORA27	snR42	S_pombe_snR42
CL00033	snopsi18S-841	SNORA28	snR80
CL00034	SNORA35	SNORA50	SNORA54	SNORA76
CL00035	SNORA36	snR36	snR44	S_pombe_snR36
CL00036	SNORA44	SNORA58	snR161	snR9
CL00037	SNORA48	snR86
CL00038	snoR80	snoR98	SNORA18	SNORA52	SNORA66	snR49	snR5	S_pombe_snR5
CL00039	SNORA56	snR8
CL00040	SNORA62	snR3	snR82	S_pombe_snR3
CL00041	SNORA64	snR37
CL00042	snoR2	SNORA65	snR34
CL00043	SNORA74	snR191
CL00044	SNORD12	snR190
CL00045	SNORD15	snoZ5	snR13	snR75
CL00046	SNORD16	snR87
CL00047	SNORD18	snoU18
CL00048	SNORD19	SNORD19B
CL00049	SNORD25	snR56
CL00050	SNORD26	SNORD81
CL00051	snoR69Y	SNORD29	SNORD36	SNORD38	SNORD78	snosnR61	snosnR69	snosnR71	snoU36a	snoZ223	snR47
CL00052	SNORD30	snoU30
CL00053	snoR35	SNORD31	snoZ17	snR67
CL00054	snoMe18S-Um1356	snoMe28S-Am982	SNORD33	SNORD51	snosnR55	snoZ196	snR39	snR40
CL00055	SNORD34	snR62
CL00056	SNORD35	snoZ161_228	snR73
CL00057	snoR77Y	SNORD39	SNORD65	snoZ101	snoZ7	snR77
CL00058	SNORD41	SNORD57	snR51
CL00059	SNORD43	snR70
CL00060	SNORD44	snoZ102_R77
CL00061	SNORD46	snR63
CL00062	SNORD49	snoU49	snoZ112
CL00063	SNORD52	SNORD53_SNORD92	snoZ157	snR78
CL00064	SNORD58	SNORD99
CL00065	SNORD59	snosnR54	snoZ159
CL00066	sn2903	snoMe28S-G3255	snoR1	SNORD60	snosnR48
CL00067	SNORD125	SNORD61	snoU61
CL00068	snoR41	SNORD62	snR41
CL00069	snoMe28S-Cm2645	snoR44_J54	SNORD109A	SNORD112	SNORD113	SNORD64	SNORD74	SNORD79	snosnR64
CL00070	Afu_263	SNORD77	snosnR60_Z15
CL00071	snoR118	SNORD88	snR76
CL00072	SNORD2	SNORD96
CL00073	snoR30	SNORD100
CL00074	snoR60	SNORD101
CL00075	SNORD105	snoU105B
CL00076	snoR14	SNORD110
CL00077	SNORND104	snR58
CL00078	SNORA73	snR30
CL00079	snoR27	snR68
CL00080	snoR53	snoR53Y
CL00081	snoU13	snoZ279_R105_R108
CL00082	snoU85	snoU89
CL00083	mir-11	mir-6
CL00084	mir-3	mir-318
CL00085	mir-15	mir-16
CL00086	mir-28	mir-708
CL00087	mir-34	mir-449
CL00088	mir-35	mir-35_2	mir-36	mir-36_2	mir-39	mir-42
CL00089	mir-190	mir-50
CL00090	mir-73	mir-74
CL00091	bantam	mir-81
CL00092	mir-137	mir-234
CL00093	mir-182	mir-183	mir-228	mir-263
CL00094	mir-216	mir-283
CL00095	mir-279	mir-996
CL00096	mir-290	mir-302	miR-430	mir-515
CL00097	mir-BART1	mir-BART3
CL00098	MIR169_2	MIR169_3	MIR169_5	MIR169_6	MIR169_7	MIR169_8
CL00099	MIR171_1	MIR171_2
CL00100	ACEA_U3	Fungi_U3	Plant_U3	U3
CL00101	AdoCbl	AdoCbl-II	Cobalamin
CL00102	group-II-D1D4-1	group-II-D1D4-2	group-II-D1D4-3	group-II-D1D4-4	group-II-D1D4-5	group-II-D1D4-6	group-II-D1D4-7
CL00103	SNORD11	SNORD11B
CL00104	AHBV_epsilon	HBV_epsilon
CL00105	SdsR_RyeB	SraC_RyeA
CL00106	CsrB	CsrC	McaS	PrrB_RsmZ	RsmV	RsmW	rsmX	RsmY	TwoAYGGAY	Ysr186_sR026_CsrC
CL00108	ar15	suhB
CL00110	mir-19	mir-363
CL00113	5S_rRNA	mtPerm-5S
CL00114	LhrC	rli22	rli33
CL00115	DUF805	DUF805b
CL00116	aCoV-5UTR	bCoV-5UTR	dCoV-5UTR	gCoV-5UTR	Sarbecovirus-5UTR
CL00117	aCoV-3UTR	bCoV-3UTR	Corona_pk3	dCoV-3UTR	gCoV-3UTR	s2m	Sarbecovirus-3UTR
CL00118	L31-Actinobacteria	L31-Coriobacteria	L31-Corynebacteriaceae	L31-Firmicutes	L31-Gammaproteobacteria
CL00119	S4-Bacteroidia	S4-Clostridia	S4-Flavobacteria	S4-Fusobacteriales
CL00120	twister-P1	twister-P3	Twister-P5
CL00121	S15	S15-Flavobacteria	S15-Halobacteria	S15-Methanomicrobia
CL00122	NiCo	NiCo-AnGGAG
CL00123	2dG-I	2dG-II	Purine
CL00124	TD-1	TD-2
CL00125	Glycine	glycine-GGAnGA
CL00126	c-di-GMP-I	c-di-GMP-I-GGC	c-di-GMP-I-UAU
CL00127	c-di-GMP-II	c-di-GMP-II-GAG	c-di-GMP-II-GCG
CL00128	C4	c4-2	c4-a1b1
CL00129	cHP	DENV_SLA	Flavivirus-5UTR
CL00131	Flavi_ISFV_repeat_Ra	Flavi_ISFV_repeat_Ra_Rb	Flavi_ISFV_repeat_Rb
CL00132	snoPyro_CD	snoR9	sR47	sR60
CL00133	mir-31	mir-72
CL00135	mir-251	mir-252
CL00136	mir-465	mir-507	mir-509	mir-513	mir-743	mir-890	mir-8908
CL00137	mir-310	mir-92
CL00138	mir-236	mir-8
CL00139	MIR2863	MIR5070
CL00140	mir-506	mir-511
CL00141	mir-H20	mir-H7
CL00142	mir-2733	mir-2851	mir-64
CL00143	MIR827	MIR827_2
CL00144	MIR4371	MIR4387
CL00145	MIR1520	MIR4372
CL00146	mir-8791	mir-8799
CL00147	mir-574	mir-9201
CL00148	let-7	mir-3596
CL00149	mir-1197	mir-154	mir-329	mir-3578	mir-368	mir-379	mir-485	mir-889
CL00150	MIR162_1	MIR162_2
CL00151	Pestivirus-3UTR-SLIII-scrofae	Pestivirus-3UTR-SLIV-SLII-BVDV-1	Pestivirus-3UTR-SLIV-SLIII-SLII-CSFV
CL00152	Plastid-psbJ-1	Plastid-psbJ-2	Plastid-psbJ-3	Plastid-psbJ-4	Plastid-psbJ-5
CL00153	Plastid-psbA-1	Plastid-psbA-2	Plastid-psbA-3	Plastid-psbA-4
CL00154	Plastid-ccsA-ndhD-1
CL00155	Plastid-ndhA-ndhB-1	Plastid-ndhA-ndhB-2	Plastid-ndhA-ndhB-3'''


