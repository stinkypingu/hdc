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

posFileName = f"datasets/unbiased/{TARGET}/active_T.smi"
negFileName = f"datasets/unbiased/{TARGET}/inactive_T_large.smi"


pos_features = np.load('saved_sparse_binary/MAPK1_features_active_T.npy')
neg_features = np.load('saved_sparse_binary/MAPK1_features_inactive_T.npy')
projection_matrix = np.load('saved_sparse_binary/MAPK1_projection_matrix.npy')

def hash(feature_matrix, projection_matrix, k=1, as_set=True):
    print(f'hashing...')
    start_time = time.time()
    
    hashset = []

    for feature in feature_matrix:
        mask = projection_matrix[feature != 0]

        hash = np.sum(mask, axis=0)

        binary_hash = np.zeros_like(hash)
        top_indices = np.argsort(hash)[-k:]  # Indices of the k highest elements
        binary_hash[top_indices] = 1
        
        hashset.append(binary_hash)
    hashes = np.vstack(hashset)

    print(f'hashes shape (before unique): {hashes.shape}')
    
    hashes = np.unique(hashes, axis=0)

    failed_hashes = 0

    print(hashes[0, :50])
    print(np.sum(hashes[0]))

    #hashes2 = feature_matrix @ projection_matrix #~550s
    end_time = time.time()
    print(f'hashes shape: {hashes.shape}')
    print(f'elapsed time: {(end_time - start_time):.6f}\n')
    
    return hashes


start_time = time.time()

pos_train_hashes = hash(pos_features, projection_matrix)
neg_train_hashes = hash(neg_features, projection_matrix)






#validate
print("\nVALIDATION")
print('------------')
TARGET="MAPK1"

pos_features = np.load('saved_sparse_binary/MAPK1_features_active_V.npy')
neg_features = np.load('saved_sparse_binary/MAPK1_features_inactive_V.npy')[:77, :]

pos_test_hashes = hash(pos_features, projection_matrix, as_set=False)
neg_test_hashes = hash(neg_features, projection_matrix, as_set=False)



end_time = time.time()
print(f'total elapsed time: {(end_time - start_time):.6f}')






