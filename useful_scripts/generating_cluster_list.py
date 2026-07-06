import re
import math

NM_list = [2]
def parse_cluster(clus):
    pat = r"([A-Z][a-z]?)(\d*)"
    matches = re.findall(pat, clus)
    ele_dict = {}

    for match in matches:
        ele = match[0]
        count = int(match[1]) if match[1] else 1
        if ele in ele_dict.keys():
            ele_dict[ele] += count
        else:
            ele_dict[ele] = count
    return ele_dict

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
TM = ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
      "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
      "La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"]
LG = ["C", "N", "O", "S"]
def charge(clus):
    if clus[-1] == '-':
        return -1
    elif clus[-1] == '+':
        return 1
    else:
        return 0

def electrons(clus):
    e = charge(clus)
    for ele in parse_cluster(clus).keys():
        e += ANDict[ele] * parse_cluster(clus)[ele]
    return e

def can_add(Array, type, num):
    next_arrays = []
    for a_combo in Array:
        pend = a_combo[:]
        if len(a_combo) < type:
            if len(a_combo) == type - 1:
                pend.append(num - sum(a_combo))
                next_arrays.append(pend)
            else:
                for k in range(1, num - sum(a_combo) - (type - len(a_combo)) + 2):
                    pend.append(k)
                    next_arrays.append(pend)
                    pend = a_combo[:]
        else:
            next_arrays.append(pend)
    return next_arrays

def plate_method(type, num):
    ARRAY = []
    if num == 0:
        return [[]]
    elif type == 1:
        ARRAY.append([num])
    else:
        for j in range(1, num - type + 2):
            ARRAY.append([j])
        for idx in range(type - 1):
            ARRAY = can_add(ARRAY, type, num)
    return ARRAY

def add_sth(ARRAY, type):
    new_ARRAY = []
    for an_array in ARRAY:
        the_array = an_array[:]
        for i, ele in enumerate(type):
            if len(the_array) == 0:
                the_array.append(ele)
                new_ARRAY.append(the_array)
                the_array = an_array[:]
            elif ele == the_array[-1]:
                k = i
                for idx in range(k + 1, len(type)):
                    the_array.append(type[idx])
                    new_ARRAY.append(the_array)
                    the_array = an_array[:]
    return new_ARRAY

def NAME(type, type_no):
     ALL = [[]]
     for _ in range(type_no):
         ALL = add_sth(ALL, type)
     return ALL

def atom_with_no(name_list, no_list):
    comp = ''
    for idx in range(len(name_list)):
        comp = comp + name_list[idx] + str(no_list[idx])
    return comp


def do_round(clus, pred_float):
    if electrons(clus) % 2 == 0:
        phys = 1
    else:
        phys = 2
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

g = open("list.txt", 'w')
for a1 in NM_list:
    for a2 in range(1, a1 + 1):
        for b1 in range(0, a1 + 1):
            if b1 == 0:
                TYPE_NO = [0]
            else:
                TYPE_NO = range(1, min(b1 + 1, len(LG) + 1))
            for b2 in TYPE_NO:
                M_S = NAME(TM, a2)
                M_N = plate_method(a2, a1)
                L_S = NAME(LG, b2)
                L_N = plate_method(b2, b1)

                for I1 in M_S:
                    for J1 in M_N:
                        m_str = atom_with_no(I1, J1)
                        for I2 in L_S:
                            for J2 in L_N:
                                l_str = atom_with_no(I2, J2)

                                comp = m_str + l_str
                                for q in ['', '+', '-']:
                                    clus = comp + q
                                    g.write(clus + '\n')
g.close()











                                


