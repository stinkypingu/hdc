import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pandas as pd
import HD_library as hd
import algorithms as algo
from sklearn.model_selection import train_test_split

np.random.seed(42)
np.set_printoptions(threshold=np.inf, linewidth=1000, precision=5)


HDV_DIM = 10000
MAX_FEATURES = 1024

RADIUS = 2

'''TARGET="MAPK1"

posFileName = f"datasets/unbiased/{TARGET}/active_T.smi"
negFileName = f"datasets/unbiased/{TARGET}/inactive_T_small.smi"'''



pos_encodings = np.load('saved/pos_encodings_MAPK1.npy')
neg_encodings = np.load('saved/neg_encodings_MAPK1_large.npy')

#AMs
am_pos, am_neg = algo.calculate_AMs(pos_encodings, neg_encodings)


def check_closer_to(vec, am1, am2):
    return 1 if hd.cos_sim(vec, am1) > hd.cos_sim(vec, am2) else 0

for i in range(10):
    print('epoch', i+1)

    new_am_pos = am_pos
    new_am_neg = am_neg
    
    print('TRAIN')
    c = 0
    for row in pos_encodings:
        pred = check_closer_to(row, am_pos, am_neg)
        if pred == 1:
            c += 1
        else:
            new_am_pos += row
            new_am_neg -= row
    print('pos acc', c / pos_encodings.shape[0])

    c = 0
    for row in neg_encodings:
        pred = check_closer_to(row, am_pos, am_neg)
        if pred == 0:
            c += 1
        else:
            new_am_pos -= row
            new_am_neg += row
    print('neg acc', c / neg_encodings.shape[0])


    print('VALIDATION')
    v_pos_encodings = np.load('saved/pos_encodings_MAPK1_V.npy')
    v_neg_encodings = np.load('saved/neg_encodings_MAPK1_V.npy')
    c = 0
    for row in v_pos_encodings:
        pred = check_closer_to(row, am_pos, am_neg)
        if pred == 1:
            c += 1
    print('pos acc', c / v_pos_encodings.shape[0])

    c = 0
    for row in v_neg_encodings:
        pred = check_closer_to(row, am_pos, am_neg)
        if pred == 0:
            c += 1
    print('neg acc', c / v_neg_encodings.shape[0])

    am_pos = new_am_pos
    am_neg = new_am_neg




