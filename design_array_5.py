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


#pos_features = np.load('saved/pos_features_MAPK1.npy')
#neg_features = np.load('saved/neg_features_MAPK1_large.npy')
#projection_matrix = np.load('saved/projection_matrix_MAPK1.npy')

pos_features = np.load('saved2/MAPK1_features_active_T.npy')
neg_features = np.load('saved2/MAPK1_features_inactive_T.npy')
projection_matrix = np.load('saved2/MAPK1_projection_matrix.npy')




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


def split_matrix(matrix, splits):
    rows_per_split = matrix.shape[0] // splits
    split_matrices = []
    
    for i in range(splits):
        start_idx = i * rows_per_split
        end_idx = (i + 1) * rows_per_split if i < splits - 1 else matrix.shape[0]  # Handle last split to ensure all rows are included
        split_matrices.append(matrix[start_idx:end_idx])    
    return split_matrices


start_time = time.time()

EPOCHS=3
SPLITS=1

pos_features_matrices = split_matrix(pos_features, SPLITS)
neg_features_matrices = split_matrix(neg_features, SPLITS)

for i in range(3):
    print(f'EPOCH: {i+1}/{EPOCHS}')
    print('------------')

    pos_am_list = []
    neg_am_list = []

    combined_mask = np.ones_like(projection_matrix, dtype=bool)
    combined_diff = np.zeros_like(projection_matrix)

    for j in range(SPLITS):
        print(f'SPLIT: {j+1}/{SPLITS}')
        print('------------')

        #encode
        pos_encodings, pos_inc_matrix, pos_dec_matrix = encode(pos_features_matrices[j], projection_matrix)
        neg_encodings, neg_inc_matrix, neg_dec_matrix = encode(neg_features_matrices[j], projection_matrix)

        pos_am, neg_am = algo.calculate_AMs(pos_encodings, neg_encodings)
        pos_am_list.append(pos_am)
        neg_am_list.append(neg_am)
        #print(np.vstack((pos_am[:10], neg_am[:10])))
    
        #results for train
        print(f'RESULTS FOR EPOCH {i+1} SPLIT {j+1}')
        out = algo.make_predictions(pos_encodings, pos_am, neg_am)
        print(f'pos acc: {np.sum(out) / out.shape[0] * 100 :.2f}%, {np.sum(out)}/{out.shape[0]}')

        out = algo.make_predictions(neg_encodings, pos_am, neg_am)
        print(f'neg acc: {(out.shape[0] - np.sum(out)) / out.shape[0] * 100:.2f}%, {(out.shape[0] - np.sum(out))}/{out.shape[0]}')

        out = algo.rocauc(pos_encodings, neg_encodings, pos_am, neg_am)
        print(f'roc_auc: {out}')
        print()

        #flip 
        pdiffy = pos_inc_matrix - pos_dec_matrix
        ndiffy = neg_inc_matrix - neg_dec_matrix

        mask = (pdiffy < 0) & (ndiffy > 0) #all items where flipping would make pos_am more positive and neg_am more negative
        combined_mask &= mask
        
        diff = np.abs(pdiffy - ndiffy)
        combined_diff += diff

    final_mask = np.zeros_like(projection_matrix, dtype=bool)
    for col in range(projection_matrix.shape[1]):
        valid_rows = np.where(combined_mask[:, col])[0]
        if valid_rows.size > 0:
            max_row = valid_rows[np.argmax(combined_diff[valid_rows, col])]
            final_mask[max_row, col] = True

    num_flips = np.sum(final_mask)
    print(f'flips: {num_flips}')
    projection_matrix[final_mask] *= -1




#validate
print("\nVALIDATION")
print('------------')
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
def vectors_are_different(vector_list):
    # Iterate over each pair of vectors
    for i in range(len(vector_list)):
        for j in range(i + 1, len(vector_list)):
            if np.array_equal(vector_list[i], vector_list[j]):
                return False  # Found two identical vectors
    return True  # All vectors are different

# Check if vectors are different
are_different = vectors_are_different(pos_am_list)
print("Vectors are different:", are_different)
print(len(pos_am_list))
print(len(neg_am_list))

out = algo.make_clustered_predictions(pos_encodings, pos_am_list, neg_am_list)
print(f'{np.sum(out) / out.shape[0] * 100 :.2f}%, {np.sum(out)}/{out.shape[0]}')

out = algo.make_clustered_predictions(neg_encodings, pos_am_list, neg_am_list)
print(f'{(out.shape[0] - np.sum(out)) / out.shape[0] * 100:.2f}%, {(out.shape[0] - np.sum(out))}/{out.shape[0]}')

out = algo.clustered_metrics(pos_encodings, neg_encodings, pos_am_list, neg_am_list)
print(f'roc_auc: {out}')



end_time = time.time()
print(f'total elapsed time: {(end_time - start_time):.6f}')









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


    
   







