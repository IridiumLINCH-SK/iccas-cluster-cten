import re
ANDict = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5,
          "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
          "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
          "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
          "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25,
          "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
          "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35,
          "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40,
          "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45,
          "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
          "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55,
          "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60,
          "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65,
          "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
          "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75,
          "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
          "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85,
          "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
          "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95,
          "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100,
          "Md": 101, "No": 102, "Lr": 103, "Rh": 104, "Db": 105,
          "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110,
          "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115,
          "Lv": 116, "Ts": 117, "Og": 118}
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

def charge(clus):
    if clus[-1] == '-':
        return -1
    elif clus[-1] == '+':
        return 1
    else:
        return 0
def electron(clus):
    return sum([ANDict[ele] for ele in parse_cluster(clus).keys()]) - charge(clus)

def do_round(line):
    clus = line.split(',')[0]
    phys = electron(clus) % 2 + 1
    pred_float = float(line.split(",")[1])
    pred = round(pred_float)
    if (pred - phys) % 2 == 1:
        if abs(pred + 1 - pred_float) < abs(pred - 1 - pred_float):
            pred = pred + 1
        else:
            pred = pred - 1

    if pred <= 0:
        if phys % 2 == 0:
            pred = 2
        else:
            pred = 1
    return pred


if __name__ == "__main__":
    f = open("fnsm.csv")
    title = f.readline()
    line = f.readline()
    g = open("phys_sm.csv", 'w')
    g.write(title)
    while line:
        clus = line.split(',')[0]
        g.write("{},{}\n".format(clus, do_round(line)))
        line = f.readline()

