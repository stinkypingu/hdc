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

    failed_hashes = 0
    for feature in feature_matrix:
        mask = projection_matrix[feature != 0]

        hash = np.sum(mask, axis=0)

        binary_hash = np.zeros_like(hash)
        top_indices = np.argsort(hash)[-k:]  # Indices of the k highest elements
        binary_hash[top_indices] = 1
        #hash = np.where(hash > 2, 1, 0) 
        
        if (np.sum(hash) == 0):
            failed_hashes += 1

        hashset.append(binary_hash)

    print(f'failed hashes: {failed_hashes}')
    hashes = np.vstack(hashset)

    if as_set:
        print(f'hashes shape: {hashes.shape}')
        print(f'returning unique hashes')
        hashes = np.unique(hashes, axis=0)
        #print(np.sum(hashes, axis=1))
    else:
        print(f'returning all hashes')

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


pos_count = 0
for hash in pos_test_hashes:
    best_pos_dist = np.min(np.linalg.norm(pos_train_hashes - hash, axis=1))
    best_neg_dist = np.min(np.linalg.norm(neg_train_hashes - hash, axis=1))

    if (best_pos_dist < best_neg_dist):
        pos_count += 1
    
    pred1 = 0.5 + (0.25 * (best_pos_dist - best_neg_dist))
    exp1 = np.ones_like(pred1)
print(f'pos acc: {pos_count / pos_test_hashes.shape[0] * 100 :.2f}%, {pos_count}/{pos_test_hashes.shape[0]}')

neg_count = 0
for hash in neg_test_hashes:
    best_pos_dist = np.min(np.linalg.norm(pos_train_hashes - hash, axis=1))
    best_neg_dist = np.min(np.linalg.norm(neg_train_hashes - hash, axis=1))
    
    if (best_neg_dist < best_pos_dist):
        neg_count += 1

    pred2 = 0.5 + (0.25 * (best_pos_dist - best_neg_dist))
    exp2 = np.zeros_like(pred2)
print(f'neg acc: {neg_count / neg_test_hashes.shape[0] * 100 :.2f}%, {neg_count}/{neg_test_hashes.shape[0]}')

pred = np.hstack((pred1, pred2))
exp = np.hstack((exp1, exp2))

print(f'roc_auc: {roc_auc_score(exp, pred)}')



end_time = time.time()
print(f'total elapsed time: {(end_time - start_time):.6f}')






