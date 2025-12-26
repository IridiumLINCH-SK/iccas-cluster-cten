import os
import re
import pandas as pd

#### Necessary Functions
def parse_cluster(cluster):
    pattern = r"([A-Z][a-z]?)(\d*)"
    matches = re.findall(pattern, cluster)
    elements_dict = {}
    for match in matches:
        element = match[0]
        count = int(match[1]) if match[1] else 1
        if element not in elements_dict.keys():
            elements_dict[element] = 0
            elements_dict[element] += count
        else:
            elements_dict[element] += count
    return elements_dict

def atom_count(cluster):
    return sum(parse_cluster(cluster).values())

def same_or_not(clus1, clus2):
    SAME = True
    dict1 = parse_cluster(clus1)
    dict2 = parse_cluster(clus2)
    for ele in dict1.keys():
        if ele not in dict2.keys():
            SAME = False
            return SAME
    for ele in dict2.keys():
        if ele not in dict1.keys():
            SAME = False
            return SAME
    for ele in dict1.keys():
        if dict1[ele] != dict2[ele]:
            SAME = False
            return SAME
    return SAME


NM = ['C', 'N', 'O', 'S']
TM = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
	'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
	'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg']
#### Functions end.

#### Load atom and molecular properties.
ap_df = pd.read_excel("atomprop.xlsx", index_col=0)
#print(list(ap_df.columns))
#### Atom and molecular properties loaded.

#### Extract info from TRAIN, VAL and TEST.
old_key_header = ['cluster', 'FEPA']
f1 = open('train_set.csv', 'r').read().splitlines()
f2 = open('val_set.csv', 'r').read().splitlines()
f3 = open('test_set.csv', 'r').read().splitlines()

if f1[0] != f2[0] or f2[0] != f3[0]:
    print("Different table headers detected, are TRAIN, VAL AND TEST datasets really remained original?")
    quit()
else:
    title = f1[0]
    headers = title.split(',')

df1 = pd.read_csv("train_set.csv")
df2 = pd.read_csv("val_set.csv")
df3 = pd.read_csv("test_set.csv")
g1 = open('train_aligned.csv', 'w')
g2 = open('val_aligned.csv', 'w')
g3 = open('test_aligned.csv', 'w')

Title = 'cluster,'
#for sth in list(mp_df.columns):
#    Title = Title + sth + ','
for an in range(1, 9):
    for sth in list(ap_df.columns):
        Title = Title + 'A{}_{},'.format(str(an), sth)
Title = Title + 'Q,FEPA\n'

g1.write(Title)
g2.write(Title)
g3.write(Title)

#### write TRAIN ####
for idx in range(0, len(df1)):
    clus = df1['cluster'][idx]
    lgk1 = df1['FEPA'][idx]
    if atom_count(clus) <= 8:
        atom_feat_str = ''
        comp_dict = parse_cluster(clus)
        for ele in TM + NM:
            if ele in comp_dict.keys():
                atom_vec = ap_df.loc[ele]
                for _ in range(0, comp_dict[ele]):
                    for k in range(0, len(atom_vec)):
                        atom_feat_str = atom_feat_str + str(atom_vec[k]) + ','
        for _ in range(0, 8 - atom_count(clus)):
            for k in range(0, len(atom_vec)):
                atom_feat_str = atom_feat_str + '0.0,'
        if clus[-1] == '-':
            q = -1
        elif clus[-1] == '+':
            q = 1
        else:
            q = 0
        g1.write('{},{}{},{}\n'.format(clus, atom_feat_str, q, lgk1))
g1.close()
#### write TRAIN finished ####


#### write VAL ####

for idx in range(0, len(df2)):
    clus = df2['cluster'][idx]
    lgk1 = df2['FEPA'][idx]
    if atom_count(clus) <= 8:
        atom_feat_str = ''
        comp_dict = parse_cluster(clus)
        for ele in TM + NM:
            if ele in comp_dict.keys():
                atom_vec = ap_df.loc[ele]
                for _ in range(0, comp_dict[ele]):
                    for k in range(0, len(atom_vec)):
                        atom_feat_str = atom_feat_str + str(atom_vec[k]) + ','
        for _ in range(0, 8 - atom_count(clus)):
            for k in range(0, len(atom_vec)):
                atom_feat_str = atom_feat_str + '0.0,'
        if clus[-1] == '-':
            q = -1
        elif clus[-1] == '+':
            q = 1
        else:
            q = 0
        g2.write('{},{}{},{}\n'.format(clus, atom_feat_str, q, lgk1))
g2.close()
#### write VAL finished ####

for idx in range(0, len(df3)):
    clus = df3['cluster'][idx]
    lgk1 = df3['FEPA'][idx]
    if atom_count(clus) <= 8:
        atom_feat_str = ''
        comp_dict = parse_cluster(clus)
        for ele in TM + NM:
            if ele in comp_dict.keys():
                atom_vec = ap_df.loc[ele]
                for _ in range(0, comp_dict[ele]):
                    for k in range(0, len(atom_vec)):
                        atom_feat_str = atom_feat_str + str(atom_vec[k]) + ','
        for _ in range(0, 8 - atom_count(clus)):
            for k in range(0, len(atom_vec)):
                atom_feat_str = atom_feat_str + '0.0,'
        if clus[-1] == '-':
            q = -1
        elif clus[-1] == '+':
            q = 1
        else:
            q = 0
        g3.write('{},{}{},{}\n'.format(clus, atom_feat_str, q, lgk1))
g3.close()
#f = open('Features_table_raw.csv', 'w')

#f.close()
