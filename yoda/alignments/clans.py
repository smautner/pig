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
    for i,e in enumerate(clans.split(f'\n')):
        line = e.split()
        d[line[0]] = line[1:]
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



def merge_clans(mode):
    rf15 = read_clans(clans_rfam15, fixrf14=False)
    rf14 = read_clans(clans, fixrf14=True)
    'basically 14 determines whats in the train and test set'
    if mode == 'train':
        # return cl from 14
        return rf14

    elif mode == 'test':
        # rf15 but we remove stuff that is already in 14...
        burned = []
        for k in list(rf15.keys()):
            if k not in rf14:
                continue
            # else we need to ckeck the values...
            for v in list(rf15[k]):
                if v in rf14[k]:
                    burned.append(v)
        return rf15, set(burned)


    else:
        assert False
    return rf15


def stats():
    def show(d):
        print(f"{len(d)=} {sum([len(v) for v in d.values()])=}")
    rf15 = read_clans(clans_rfam15, fixrf14=False)
    print(f"rf15 total")
    show(rf15)

    print(f"train")
    d = merge_clans(mode='train')
    show(d)

    d = merge_clans(mode='test')
    print(f"test")
    show(d)


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
clans_rfam15 = '''CL00110	mir-19	mir-363
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


