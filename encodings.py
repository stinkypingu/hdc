import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pandas as pd
import HD_library as hd
import algorithms as algo
from sklearn.model_selection import train_test_split
from scipy.sparse import random as sparse_random
from scipy.sparse import coo_matrix

np.random.seed(42)
np.set_printoptions(threshold=np.inf, linewidth=1000, precision=5)


HDV_DIM = 4000
MAX_FEATURES = 1024

RADIUS = 2

TARGET="MAPK1"
    
posFileName = f"datasets/unbiased/{TARGET}/active_V.smi"
negFileName = f"datasets/unbiased/{TARGET}/inactive_V.smi"

#molecule data from files
pos_df = algo.get_dataframe(posFileName, actual=1)
neg_df = algo.get_dataframe(negFileName, actual=0)

#features of each molecule
pos_features = algo.get_features_matrix(pos_df, RADIUS, MAX_FEATURES)
neg_features = algo.get_features_matrix(neg_df, RADIUS, MAX_FEATURES)

#generate random HDV projection matrix
#projection_matrix = algo.get_random_matrix(MAX_FEATURES, HDV_DIM)
#projection_matrix = np.load('saved/projection_matrix_MAPK1.npy')
projection_matrix = np.zeros((MAX_FEATURES, HDV_DIM), dtype=int)
for row in range(MAX_FEATURES):
    col_indices = np.random.choice(HDV_DIM, int(HDV_DIM*0.05), replace=False) #10% dense
    projection_matrix[row, col_indices] = 1


# Check the result (optional)
#build encodings of each molecule
#pos_encodings = algo.get_encodings_batched(pos_features, projection_matrix, 100)
#neg_encodings = algo.get_encodings_batched(neg_features, projection_matrix, 100)

np.save('saved_sparse_binary/MAPK1_projection_matrix.npy', projection_matrix)
np.save('saved_sparse_binary/MAPK1_features_active_V.npy', pos_features)
np.save('saved_sparse_binary/MAPK1_features_inactive_V.npy', neg_features)
