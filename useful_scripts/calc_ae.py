import os
import re

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

def atom_no(clus):
    return sum(parse_cluster(clus).values())

NM = ['C', 'N', 'O', 'S']
TM = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
	'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
	'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg']

Info_lines = open("EZPE.csv", 'r').read().splitlines()
ATOMS = {}
for line in Info_lines[1:]:
    clus = line.split(',')[0]
    EZPE = float(line.split(',')[1])
    if atom_no(clus) == 1:
        ATOMS[clus] = EZPE

def aepa(clus, EZPE):
    if atom_no(clus) == 1:
        return 0
    ele_dict = parse_cluster(clus)
    if clus[-1] == '-':
        charge = -1
    elif clus[-1] == '+':
        charge = 1
    else:
        charge = 0

    form_E = EZPE
    for ele in list(ele_dict.keys()):
        neu_atom_E = ATOMS[ele + '1']
        form_E = form_E - ele_dict[ele] * neu_atom_E
    if charge == 1:
        atom_ips = {}
        for ele in list(ele_dict.keys()):
            neu_atom_E = ATOMS[ele + '1']
            cation_E = ATOMS[ele + '1+']
            atom_ips[ele] = cation_E - neu_atom_E
            
        form_E -= min(atom_ips.values())
    elif charge == -1:
        atom_eas = {}
        for ele in list(ele_dict.keys()):
            neu_atom_E = ATOMS[ele + '1']
            anion_E = ATOMS[ele + '1-']
            atom_eas[ele] = neu_atom_E - anion_E

        form_E += max(0, max(atom_eas.values()))
    AEPA =  - 27.2114 * form_E / sum(parse_cluster(clus).values())
    return AEPA

g = open("AEPA.csv", 'w')
g.write("cluster,AEPA\n")
for line in Info_lines[1:]:
    clus, EZPE = line.split(',')
    EZPE = float(EZPE)
    AEPA = aepa(clus, EZPE)
    g.write("{},{:.4f}\n".format(clus, AEPA))
g.close()
