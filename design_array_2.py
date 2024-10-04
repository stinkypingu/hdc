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


HDV_DIM = 100
MAX_FEATURES = 1024

RADIUS = 2

TARGET="MAPK1"

posFileName = f"datasets/unbiased/{TARGET}/active_T.smi"
negFileName = f"datasets/unbiased/{TARGET}/inactive_T_large.smi"

#molecule data from files
pos_df = algo.get_dataframe(posFileName, actual=1)
neg_df = algo.get_dataframe(negFileName, actual=0)

#features of each molecule
pos_features = algo.get_features_matrix(pos_df, RADIUS, MAX_FEATURES)
neg_features = algo.get_features_matrix(neg_df, RADIUS, MAX_FEATURES)

#generate random HDV projection matrix
projection_matrix = algo.get_random_matrix(MAX_FEATURES, HDV_DIM)


#cooccurrence matrices
pos_cooc = algo.get_cooccurrence_matrix(pos_features, 1000)
print(len(pos_df))
print(pos_cooc[:6, :6])



neg_cooc = algo.get_cooccurrence_matrix(neg_features, 1000)
print(len(neg_df))
print(neg_cooc[:6, :6])






posFileName = f"datasets/unbiased/{TARGET}/active_V.smi"
negFileName = f"datasets/unbiased/{TARGET}/inactive_V.smi"

#molecule data from files
pos_df = algo.get_dataframe(posFileName, actual=1)
neg_df = algo.get_dataframe(negFileName, actual=0)

#features of each molecule
pos_features = algo.get_features_matrix(pos_df, RADIUS, MAX_FEATURES)
neg_features = algo.get_features_matrix(neg_df, RADIUS, MAX_FEATURES)

c = 0
for x in pos_features:
    dot_product = np.dot(norm_pos_cooc, x)
    matrix_norms = np.linalg.norm(norm_pos_cooc, axis=1)
    vector_norm = np.linalg.norm(x)

    pos_cs = dot_product / (matrix_norms * vector_norm)
    pos_cs = np.nan_to_num(pos_cs, nan=0.0)


    dot_product = np.dot(norm_neg_cooc, x)
    matrix_norms = np.linalg.norm(norm_neg_cooc, axis=1)

    neg_cs = dot_product / (matrix_norms * vector_norm)
    neg_cs = np.nan_to_num(neg_cs, nan=0.0)



    #print(np.max(pos_cs) - np.max(neg_cs))

    if (np.max(pos_cs) > np.max(neg_cs)):
        c += 1
print('pos acc:', c / pos_features.shape[0])





c = 0
for x in neg_features:
    dot_product = np.dot(norm_pos_cooc, x)
    matrix_norms = np.linalg.norm(norm_pos_cooc, axis=1)
    vector_norm = np.linalg.norm(x)

    pos_cs = dot_product / (matrix_norms * vector_norm)
    pos_cs = np.nan_to_num(pos_cs, nan=0.0)


    dot_product = np.dot(norm_neg_cooc, x)
    matrix_norms = np.linalg.norm(norm_neg_cooc, axis=1)

    neg_cs = dot_product / (matrix_norms * vector_norm)
    neg_cs = np.nan_to_num(neg_cs, nan=0.0)



    #print(np.max(neg_cs) - np.max(pos_cs))

    if (np.max(neg_cs) > np.max(pos_cs)):
        c += 1
print('neg acc:', c / neg_features.shape[0])



print()


prev_inc = None
prev_dec = None
for i in range(0):
    print('epoch ', i+1)

    #build encodings of each molecule
    pos_encodings = algo.get_encodings(pos_features, projection_matrix)
    neg_encodings = algo.get_encodings(neg_features, projection_matrix)

    #keep track of AM+ encoding +1/-1 bit count
    am_pos_inc, am_pos_dec = algo.track_usage(pos_features, pos_encodings)

    #keep track of AM- encoding +1/-1 bit count
    am_neg_inc, am_neg_dec = algo.track_usage(neg_features, neg_encodings)



    print('inc\n', am_pos_inc[:6, :6])
    #print('dec\n', am_pos_dec[:10, 3:10])

    #print(np.mean(np.abs(am_pos_inc - am_pos_dec), axis=0)[:10])
    #print(np.mean(np.abs(am_neg_inc - am_neg_dec), axis=0)[:10])
    #print(np.count_nonzero(np.abs(am_pos_inc - am_pos_dec) < 3))  

    #idx = algo.flip_bits(am_pos_inc, am_pos_dec, am_neg_inc, am_neg_dec)
    #print()


    #AMs
    am_pos, am_neg = algo.calculate_AMs(pos_encodings, neg_encodings)
    print(am_pos[:6])
    print(am_neg[:6])

    print(np.mean(np.abs(am_pos)))
    print(np.mean(np.abs(am_neg)))

    print('opposite polarities:', np.sum((am_pos * am_neg) < 0))


    #change in AM wrt bit flips
    delta_am_pos = (am_pos_inc - am_pos_dec) * -2
    delta_am_neg = (am_neg_inc - am_neg_dec) * -2
    #print(delta_am_pos[:6, :6]) 
    #print(delta_am_neg[:6, :6])



    y = algo.make_predictions_hamming(pos_encodings, am_pos, am_neg)
    print('pos acc:', np.sum(y) / y.shape[0])

    y = algo.make_predictions_hamming(neg_encodings, am_pos, am_neg)
    print('neg acc:', np.sum(y) / y.shape[0])




    #flip the bit that maximizes polarity diff in AMs
    mask = am_pos * am_neg #polarity
    potential_mask = (delta_am_pos + am_pos) * (delta_am_neg + am_neg) #polarity after potential bit flips

    flips = (mask > 0) & (potential_mask < 0)
    
    def randomly_select_true_per_column(mask):
        # Create a result mask initialized with False values
        result_mask = np.zeros_like(mask, dtype=bool)
        
        # Iterate over each column
        for col in range(mask.shape[1]):
            # Get the row indices where the mask is True in this column
            true_indices = np.where(mask[:, col])[0]
            
            if true_indices.size > 0:  # If there are True values in the column
                # Randomly select one of the True indices
                selected_index = np.random.choice(true_indices)
                # Set only the selected index to True in the result mask
                result_mask[selected_index, col] = True
            else:
                # If no True values, copy the entire column as is
                result_mask[:, col] = mask[:, col]

        return result_mask
    
    flips = randomly_select_true_per_column(flips)
    print('number of potential flips:', np.sum(flips == True))



    idx = np.unravel_index(np.argmax(flips), flips.shape)
    print('idx:', idx)
    

    flip_mask = np.ones_like(projection_matrix)
    flip_mask[idx] = -1
    #projection_matrix *= flip_mask

    projection_matrix *= np.where(flips, 1, -1)
    print()


