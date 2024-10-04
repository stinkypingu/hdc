import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import HD_library as hd
import pandas as pd
from sklearn.metrics import roc_auc_score

#returns fingerprint given a smiles string, radius, and nBits
def get_single_fingerprint(smiles, radius, max_features):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=max_features)
    npfp = np.zeros((1,), dtype=np.int32)
    DataStructs.ConvertToNumpyArray(fp, npfp)
    return npfp

#randomly generates a weight matrix
def get_random_matrix(num_features, vector_dimension):
    return np.random.choice([1, -1], size=(num_features, vector_dimension))

#use array of features present to build the encoding from a projection matrix
def get_single_encoding(features, projection_matrix):
    mask = features.astype(bool) #[0 1 0 1 0 0...]
    selected_rows = projection_matrix[mask, :]
    binded = np.prod(selected_rows, axis=0)
    return binded

#returns the binded encodings by multiplying the projected HDVs from the features. avoids using single encoding by making a new axis
def get_encodings(features_matrix, projection_matrix):
    mask = features_matrix.astype(bool)[:, :, np.newaxis] #make a new axis to collect feature HDVs
    masked_proj = np.where(mask, projection_matrix, 1) #so we don't zero out data accidentally
    encodings = np.prod(masked_proj, axis=1) #bundle
    return encodings

def get_encodings_batched(features_matrix, projection_matrix, batch_size):
    num_rows = features_matrix.shape[0]
    encodings_list = []

    # Loop over the features_matrix in batches
    for start in range(0, num_rows, batch_size):
        end = min(start + batch_size, num_rows)
        batch_features = features_matrix[start:end, :]
        
        # Perform the encoding operation on the current batch
        mask = batch_features.astype(bool)[:, :, np.newaxis]  # Add new axis
        masked_proj = np.where(mask, projection_matrix, 1)  # Mask projection matrix
        encodings = np.prod(masked_proj, axis=1)  # Bundle
        
        encodings_list.append(encodings)
        print('completed:', end)
    return np.vstack(encodings_list)

#builds the dataframe with columns 'smiles' and 'id' from a file
def get_dataframe(filename, actual=None):
    df = pd.read_csv(filename, header=None, names=['smiles'])

    extract_id = lambda s: s.split(' ', 1)[1]
    df['id'] = df['smiles'].apply(extract_id)

    # remove id number at end of row
    extract_smiles = lambda s: s.split(' ', 1)[0]    
    df['smiles'] = df['smiles'].apply(extract_smiles)

    if (actual is not None):
        df['actual'] = actual
    return df

#builds np matrix of fingerprints, where each row is a new fingerprint
def get_features_matrix(df, radius, max_features):
    function = lambda smiles: get_single_fingerprint(smiles, radius, max_features)
    df['features'] = df['smiles'].apply(function)
    
    return np.vstack(df['features'].tolist())

#for each molecule, checks its composition and accumulates feature usage into arrays
def track_usage(features_matrix, encodings_matrix):
    assert(features_matrix.shape[0] == encodings_matrix.shape[0]) #same number of molecules

    inc_contribution = np.where(encodings_matrix == 1, 1, 0)
    dec_contribution = np.ones_like(inc_contribution) - inc_contribution

    inc_matrix = features_matrix.T @ inc_contribution #cant believe this works
    dec_matrix = features_matrix.T @ dec_contribution 

    return inc_matrix, dec_matrix

#co occurrence matrix for features to features
def get_cooccurrence_matrix(features_matrix, batch_size):
    num_features = features_matrix.shape[1]

    cooccurrence = np.zeros((num_features, num_features))

    num_rows = features_matrix.shape[0]
    for start in range(0, num_rows, batch_size):
        end = min(start + batch_size, num_rows)
        batch = features_matrix[start:end]

        cooccurrence += batch.T @ batch
        print('completed:', end)
    return cooccurrence


#calculate AMs from encodingsrfew
def calculate_AMs(encodings_matrix1, encodings_matrix2):
    am1 = np.sum(encodings_matrix1, axis=0)
    am2 = np.sum(encodings_matrix2, axis=0)
    return am1, am2


#decide how badly we want to flip a bit
#x = am1, y = am2, a = delta am1, b = delta am2
def which_bits_to_flip(x, y, a, b):
    xy = np.sign(x * y)
    xy_min_dist = np.min([np.abs(x), np.abs(y)], axis=0)

    xayb = np.sign((x + a) * (y + b))
    xayb_min_dist = np.min([np.abs(x + a), np.abs(y + b)], axis=0)

    #higher => more likelihood to flip
    conditions = [
        (xy < xayb), #only xy polar -> False
        (xy > xayb), #only xyab polar -> True
        (xy == -1) & (xy_min_dist < xayb_min_dist), #both polar and xyab is further from 0 -> True
        (xy == 1) & (xy_min_dist > xayb_min_dist) #both nonpolar and xyab is closer to 0 -> True
    ]
    choices = [
        -1, #False,
        3 + xayb_min_dist, #True,
        2, #True,
        1, #True
    ]

    flip_likelihood = np.select(conditions, choices, default=-1) #default is to not flip

    '''#find the best one to flip and just flip that one
    max_index_flat = np.argmax(flip_likelihood)
    max_index_row, max_index_col = np.unravel_index(max_index_flat, flip_likelihood.shape)

    result = np.ones_like(flip_likelihood)      
    result[max_index_row, max_index_col] = -1'''
 
    #collect the best one to flip for each column/bit position
    max_indices = np.argmax(flip_likelihood, axis=0)
    result = np.ones_like(flip_likelihood)

    result[max_indices, np.arange(flip_likelihood.shape[1])] = -1
    
    return result




def flip_bits(pos_inc, pos_dec, neg_inc, neg_dec):
    pos_diff = pos_inc - pos_dec #np.abs(pos_inc - pos_dec)
    neg_diff = neg_inc - neg_dec #np.abs(neg_inc - neg_dec)

    norm_pos_diff = pos_diff / np.max(np.abs(pos_diff))
    norm_neg_diff = neg_diff / np.max(np.abs(neg_diff))

    metric = 1 - norm_pos_diff + norm_neg_diff

    index = np.unravel_index(np.argmax(metric), metric.shape)
    print(index)
    print(pos_diff[index])
    print(neg_diff[index])

    return index




def cosine_similarity(encodings_matrix, am):
    am = np.sign(am)
    am = am[np.newaxis, :]
    dot_product = np.dot(encodings_matrix, am.T).flatten() #dot products of each row with the AM
    encodings_norms = np.linalg.norm(encodings_matrix, axis=1) #norms of each encoding
    am_norm = np.linalg.norm(am) #norm of the AM
    return dot_product / (encodings_norms * am_norm)


    
def make_predictions(encodings_matrix, am1, am2):
    cos_sim_am1 = cosine_similarity(encodings_matrix, am1)
    cos_sim_am2 = cosine_similarity(encodings_matrix, am2)

    predictions = (cos_sim_am1 > cos_sim_am2).astype(int) #1 for positive class
    return predictions

def make_predictions_hamming(encodings_matrix, am1, am2):
    hamming_am1 = np.sum(encodings_matrix != np.sign(am1), axis=1)
    hamming_am2 = np.sum(encodings_matrix != np.sign(am2), axis=1)

    predictions = (hamming_am1 < hamming_am2).astype(int) #1 for positive class
    return predictions

def rocauc(pos_enc, neg_enc, am1, am2):
    pos_cos_sim_am1 = cosine_similarity(pos_enc, am1)
    pos_cos_sim_am2 = cosine_similarity(pos_enc, am2)

    pos_pred = (0.5 + (0.25 * (pos_cos_sim_am1 - pos_cos_sim_am2)))
    pos_exp = np.ones_like(pos_pred)

    #print(np.vstack((pos_cos_sim_am1[:10], pos_cos_sim_am2[:10], pos_pred[:10], pos_exp[:10])))


    neg_cos_sim_am1 = cosine_similarity(neg_enc, am1)
    neg_cos_sim_am2 = cosine_similarity(neg_enc, am2)

    neg_pred = (0.5 + (0.25 * (neg_cos_sim_am1 - neg_cos_sim_am2)))
    neg_exp = np.zeros_like(neg_pred)

    #print(np.vstack((neg_cos_sim_am1[:10], neg_cos_sim_am2[:10], neg_pred[:10], neg_exp[:10])))


    pred = np.hstack((pos_pred, neg_pred))
    exp = np.hstack((pos_exp, neg_exp))

    return roc_auc_score(exp, pred)



#works on multiple AMs
def make_clustered_predictions(encodings_matrix, pos_am_list, neg_am_list):

    pos_sims = []
    for pos_am in pos_am_list:
        similarity = cosine_similarity(encodings_matrix, pos_am)
        pos_sims.append(similarity)

    pos_sims = np.vstack(pos_sims)
    pos_max = np.max(pos_sims, axis=0)

    neg_sims = []
    for neg_am in neg_am_list:
        similarity = cosine_similarity(encodings_matrix, neg_am)
        neg_sims.append(similarity)

    neg_sims = np.vstack(neg_sims)
    neg_max = np.max(neg_sims, axis=0)


    predictions = (pos_max > neg_max).astype(int) #1 for positive class
    return predictions



def clustered_metrics(pos_enc, neg_enc, pos_am_list, neg_am_list):
    #handle positive encodings, end with best similarity for each AM list
    sim_to_pos_am = []
    for pos_am in pos_am_list:
        similarity = cosine_similarity(pos_enc, pos_am)
        sim_to_pos_am.append(similarity)
    best_sim_pos_am = np.max(np.vstack(sim_to_pos_am), axis=0)

    sim_to_neg_am = []
    for neg_am in neg_am_list:
        similarity = cosine_similarity(pos_enc, neg_am)
        sim_to_neg_am.append(similarity)
    best_sim_neg_am = np.max(np.vstack(sim_to_neg_am), axis=0)

    #results of positive encodings after using best similarities    
    pos_pred = (0.5 + (0.25 * (best_sim_pos_am - best_sim_neg_am)))
    pos_exp = np.ones_like(pos_pred)


    #handle negative encodings, end with best similarity for each AM list
    sim_to_pos_am = []
    for pos_am in pos_am_list:
        similarity = cosine_similarity(neg_enc, pos_am)
        sim_to_pos_am.append(similarity)
    best_sim_pos_am = np.max(np.vstack(sim_to_pos_am), axis=0)

    sim_to_neg_am = []
    for neg_am in neg_am_list:
        similarity = cosine_similarity(neg_enc, neg_am)
        sim_to_neg_am.append(similarity)
    best_sim_neg_am = np.max(np.vstack(sim_to_neg_am), axis=0)

    #results of negative encodings after using best similarities    
    neg_pred = (0.5 + (0.25 * (best_sim_pos_am - best_sim_neg_am)))
    neg_exp = np.zeros_like(neg_pred)


    pred = np.hstack((pos_pred, neg_pred))
    exp = np.hstack((pos_exp, neg_exp))

    return roc_auc_score(exp, pred)