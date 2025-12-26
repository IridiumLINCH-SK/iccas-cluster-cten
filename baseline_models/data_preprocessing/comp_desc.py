import os
import re
import random
import math

def parse_cluster(cluster):
    # regex (regular expression) pattern to match element symbols followed by optional numbers
    pattern = r"([A-Z][a-z]?)(\d*)"
    matches = re.findall(pattern, cluster)
    elements_dict = {}

    for match in matches:
        element = match[0]
        count = int(match[1]) if match[1] else 1  # Default to 1 if no number is present
        if element not in elements_dict.keys():
            elements_dict[element] = 0
            elements_dict[element] += count
        else:
            elements_dict[element] += count

    return elements_dict

NM = ['C', 'N', 'O', 'S']
TM = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
	'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
	'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg']

sets = ['train_set.csv', 'val_set.csv', 'test_set.csv']

for a_csv in sets:
    Info_lines = open(a_csv).read().splitlines()
    g = open('{}_comp_desc.csv'.format(a_csv.split('_')[0]), 'w')
    g.write('cluster,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Y,Zr,Nb,Mo,Tc,Ru,Rh,Pd,Ag,Cd,La,Hf,Ta,W,Re,Os,Ir,Pt,Au,Hg,C,N,O,S,Q,FEPA\n')
    for idx in range(1, len(Info_lines)):
        line = Info_lines[idx]
        clus = line.split(',')[0]
        FEPA = float(line.split(',')[1])
        g.write(clus)
        g.write(',')
        for ele in TM + NM:
            if ele not in parse_cluster(clus).keys():
                g.write('0')
            else:
                g.write(str(parse_cluster(clus)[ele]))
            g.write(',')
        if clus[-1] == '-':
            g.write('-1')
        elif clus[-1] == '+':
            g.write('1')
        else:
            g.write('0')
        g.write(',')
        g.write(str(FEPA))
        g.write('\n')
    g.close()

