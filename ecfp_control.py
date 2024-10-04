import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pandas as pd
import HD_library as hd
import algorithms as algo
from sklearn.model_selection import train_test_split
import time
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn.metrics import roc_auc_score
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

np.random.seed(42)
np.set_printoptions(threshold=np.inf, linewidth=1000, precision=5)


HDV_DIM = 10000
MAX_FEATURES = 1024

RADIUS = 2

TARGET="MAPK1"



pos_train_features = np.load('saved_sparse_binary/MAPK1_features_active_T.npy')
neg_train_features = np.load('saved_sparse_binary/MAPK1_features_inactive_T.npy')
projection_matrix = np.load('saved_sparse_binary/MAPK1_projection_matrix.npy')


start_time = time.time()






#validate
print("\nVALIDATION")
print('------------')
TARGET="MAPK1"

pos_test_features = np.load('saved_sparse_binary/MAPK1_features_active_V.npy')
neg_test_features = np.load('saved_sparse_binary/MAPK1_features_inactive_V.npy')


out = algo.clustered_metrics(pos_test_features, neg_test_features, pos_train_features, neg_train_features)
print(f'roc_auc: {out}')



end_time = time.time()
print(f'total elapsed time: {(end_time - start_time):.6f}')






