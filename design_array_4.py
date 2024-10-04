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


pos_features = np.load('saved/pos_features_MAPK1.npy')
neg_features = np.load('saved/neg_features_MAPK1_large.npy')

projection_matrix = np.load('saved/projection_matrix_MAPK1.npy')


def encode(feature_matrix, projection_matrix):
    print(f'encoding...')
    start_time = time.time()
    
    encodings_list = []

    inc_matrix = np.zeros_like(projection_matrix)
    for feature in feature_matrix:
        mask = projection_matrix[feature != 0]

        encoding = np.prod(mask, axis=0)

        inc_mask = (encoding == 1)
        inc_matrix[feature != 0] += inc_mask

        encodings_list.append(encoding)
    encodings = np.vstack(encodings_list)
    
    #reduce number of operations
    total_features = np.sum(feature_matrix, axis=0)
    dec_matrix = total_features[:, np.newaxis] - inc_matrix

    end_time = time.time()
    print(f'elapsed time: {(end_time - start_time):.6f}')
    print(f'features shape: {encodings.shape}\n')
    
    return encodings, inc_matrix, dec_matrix


for i in range(3):
    print(f'EPOCH: {i+1}\n')

    pos_encodings, pos_inc_matrix, pos_dec_matrix = encode(pos_features, projection_matrix)
    neg_encodings, neg_inc_matrix, neg_dec_matrix = encode(neg_features, projection_matrix)

    pos_am, neg_am = algo.calculate_AMs(pos_encodings, neg_encodings)
    print(np.vstack((pos_am[:10], neg_am[:10])))

    #np.set_printoptions(precision=2, suppress=True)
    #print(np.vstack((pos_am[:10] / 231, neg_am[:10] / 24761)))
    #print(np.vstack((np.sum(np.abs(pos_am[:10]) / 115), np.sum(np.abs(neg_am[:10]) / 12380))))
     
 
    #results for train
    out = algo.make_predictions(pos_encodings, pos_am, neg_am)
    print(f'{np.sum(out) / out.shape[0] * 100 :.2f}%, {np.sum(out)}/{out.shape[0]}')

    out = algo.make_predictions(neg_encodings, pos_am, neg_am)
    print(f'{(out.shape[0] - np.sum(out)) / out.shape[0] * 100:.2f}%, {(out.shape[0] - np.sum(out))}/{out.shape[0]}')

    out = algo.rocauc(pos_encodings, neg_encodings, pos_am, neg_am)
    print(f'roc_auc: {out}')



    #flip 
    pdiffy = pos_inc_matrix - pos_dec_matrix
    ndiffy = neg_inc_matrix - neg_dec_matrix

    mask = (pdiffy < 0) & (ndiffy > 0)

    differences = np.abs(pdiffy - ndiffy)
    new_mask = np.zeros_like(projection_matrix, dtype=bool)
    for col in range(projection_matrix.shape[1]):
        valid_rows = np.where(mask[:, col])[0]
        if valid_rows.size > 0:
            max_row = valid_rows[np.argmax(differences[valid_rows, col])]
            new_mask[max_row, col] = True

    projection_matrix[new_mask] *= -1




#validate
print("\nVALIDATION\n")
TARGET="MAPK1"

posFileName = f"datasets/unbiased/{TARGET}/active_V.smi"
negFileName = f"datasets/unbiased/{TARGET}/inactive_V.smi"

#molecule data from files
pos_df = algo.get_dataframe(posFileName, actual=1)
neg_df = algo.get_dataframe(negFileName, actual=0)

#features of each molecule
pos_features = algo.get_features_matrix(pos_df, RADIUS, MAX_FEATURES)
neg_features = algo.get_features_matrix(neg_df, RADIUS, MAX_FEATURES)

#encode
pos_encodings, pos_inc_matrix, pos_dec_matrix = encode(pos_features, projection_matrix)
neg_encodings, neg_inc_matrix, neg_dec_matrix = encode(neg_features, projection_matrix)

#results
out = algo.make_predictions_hamming(pos_encodings, pos_am, neg_am)
print(f'{np.sum(out) / out.shape[0] * 100 :.2f}%, {np.sum(out)}/{out.shape[0]}')

out = algo.make_predictions_hamming(neg_encodings, pos_am, neg_am)
print(f'{(out.shape[0] - np.sum(out)) / out.shape[0] * 100:.2f}%, {(out.shape[0] - np.sum(out))}/{out.shape[0]}')

out = algo.rocauc(pos_encodings, neg_encodings, pos_am, neg_am)
print(f'roc_auc: {out}')













'''X = np.vstack((pos_features, neg_features))
y = np.hstack((np.ones(pos_encodings.shape[0]), np.zeros(neg_encodings.shape[0])))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
'''



'''a = time.time()
e = encode(neg_features, projection_matrix)
print(e.shape)
b = time.time()
elapsed = b - a
print(f'elapsed: {elapsed:.6f}')


a = time.time()
neg_encodings = algo.get_encodings_batched(neg_features, projection_matrix, 100)
b = time.time()
elapsed = b - a
print(f'elapsed: {elapsed:.6f}')
'''


    
   







