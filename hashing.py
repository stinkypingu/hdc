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
import itertools

'''MAX_FEATURES = 256
HDV_DIM = 400
DENSITY = 50
K = 1
BINARY_HASHING = False'''

np.random.seed(42)
np.set_printoptions(threshold=np.inf, linewidth=1000, precision=5)


#build a projection matrix, where pn = max features, kc = projection hdv dimension, density = how many connections per kc, and seed = 42
def make_projection_matrix(pn, kc, density, seed=42):
    np.random.seed(seed)
    projection_matrix = np.zeros((pn, kc), dtype=int)
    for col in range(kc): #per column
        indices = np.random.choice(pn, density, replace=False) #10% dense
        projection_matrix[indices, col] = 1
    return projection_matrix


#hash
def hash(feature_matrix, projection_matrix, k, binary_hashing, verbose=False):

    if (verbose):        
        print(f'hashing features matrix with k={k}...')
        start_time = time.time()
    
    hashset = []
    for feature in feature_matrix:
        mask = projection_matrix[feature != 0]
        hash = np.sum(mask, axis=0)

        binary_hash = np.zeros_like(hash)
        top_indices = np.argsort(hash)[-k:]  # Indices of the k highest elements

        #hashing method
        if (binary_hashing):
            binary_hash[top_indices] = 1
        else:         
            binary_hash[top_indices] = hash[top_indices]
        
        hashset.append(binary_hash)
    hashes = np.vstack(hashset)

    if (verbose):
        print(f'hashes shape (before unique): {hashes.shape}')
    
    unique_hashes, hash_counts = np.unique(hashes, axis=0, return_counts=True)
    
    if (verbose):
        print(f'hashes shape (after unique): {unique_hashes.shape}')
        print(f'biggest bucket: {np.max(hash_counts)}')
        print(f'median bucket: {np.median(hash_counts)}')
        print(f'singletons: {np.sum(hash_counts == 1)}')

        #hashes2 = feature_matrix @ projection_matrix #~550s
        end_time = time.time()
        print(f'elapsed time: {(end_time - start_time):.6f}\n')
    
    data = (unique_hashes, hash_counts) #pack
    return data


#combines hashes from two sets together and collects how many times each class get hashed to each hash
def combine_hashes(pos_data, neg_data, verbose=False):

    pos_hashes, pos_counts = pos_data
    neg_hashes, neg_counts = neg_data

    combined_hashes = np.vstack((pos_hashes, neg_hashes))

    #collect hashes
    all_hashes, inverse = np.unique(combined_hashes, axis=0, return_inverse=True)

    all_pos_counts = np.zeros(len(all_hashes), dtype=int)
    all_neg_counts = np.zeros(len(all_hashes), dtype=int)

    pos_indices, neg_indices = np.split(inverse, [len(pos_counts)]) #mapping of indices

    all_pos_counts[pos_indices] = pos_counts
    all_neg_counts[neg_indices] = neg_counts

    assert(all_hashes.shape[0] == len(all_pos_counts)) #each hash in all_hashes should have a corresponding all_pos_count and all_neg_count
    assert(all_hashes.shape[0] == len(all_neg_counts))

    if (verbose):
        print(f'hashes shape (after merge): {all_hashes.shape}')

    all_counts = (all_pos_counts, all_neg_counts)
    return all_hashes, all_counts


#calculate at least the 5 best buckets
def train_analysis(hashes, counts, min_strong_hashes=5, verbose=False):
    pos_counts, neg_counts = counts #unpack

    percentages = pos_counts / (pos_counts + neg_counts)

    #find the best buckets based on percentage of positive class to total
    strong_hashes_indices = np.where(percentages >= 0.5)[0] #indices of the best ratiod buckets in all_unique_hashes
    if len(strong_hashes_indices) < min_strong_hashes:
        strong_hashes_indices = np.argsort(percentages)[-min_strong_hashes:]

    assert(len(strong_hashes_indices >= min_strong_hashes))

    num_hashes = len(hashes)
    all_counts = pos_counts + neg_counts
    num_singletons = np.sum(all_counts == 1)
    largest_hash = np.max(all_counts) #how many there are in the biggest bucket
    median_hash = np.median(all_counts)
    num_strong_hashes = len(strong_hashes_indices)

    if (verbose):
        #analysis print
        print(f'\nnumber of hashes: {num_hashes}')
        print(f'number of singletons: {num_singletons}')
        print(f'largest hash size: {largest_hash}')
        print(f'median hash size: {median_hash}')
        print(f'number of strong hashes: {num_strong_hashes}')

    #get the best hashes with their percentages
    strong_hashes = hashes[strong_hashes_indices]

    strong_pos_counts = pos_counts[strong_hashes_indices]
    strong_neg_counts = neg_counts[strong_hashes_indices]

    strong_counts = (strong_pos_counts, strong_neg_counts) #pack

    train_results = (num_hashes, num_singletons, largest_hash, median_hash, num_strong_hashes) #pack

    #strong hashes, their corresponding pos and neg counts, then number of buckets, number of singletons, largest bucket size, median bucket size, number of strong hashes
    return strong_hashes, strong_counts, train_results


#gets the number of molecules from the validation set that map to each of the strong hashes provided
def validate(strong_hashes, pos_data, neg_data, verbose=False):
    #unpack
    pos_hashes, pos_counts = pos_data
    neg_hashes, neg_counts = neg_data

    if (verbose):
        print(f'\nvalidation set, with {len(strong_hashes)} strong hashes:')

    result_pos_counts = []
    result_neg_counts = []

    #loop through each hash
    for hash in strong_hashes:
        #find how many from the validation set positive class have the same hash
        matching_pos_hash = np.all(pos_hashes == hash, axis=1)
        if np.any(matching_pos_hash): 
            matching_pos_idx = np.argmax(matching_pos_hash)
            matching_pos_count = pos_counts[matching_pos_idx]
        else:
            matching_pos_count = 0
            
        #find how many from the validation set negative class have the same hash
        matching_neg_hash = np.all(neg_hashes == hash, axis=1)
        if np.any(matching_neg_hash):
            matching_neg_idx = np.argmax(matching_neg_hash)
            matching_neg_count = neg_counts[matching_neg_idx]
        else:
            matching_neg_count = 0

        #store the results of how many from the validation set mapped to the strong hash
        result_pos_counts.append(matching_pos_count)
        result_neg_counts.append(matching_neg_count)

    valid_pos_counts = np.array(result_pos_counts)
    valid_neg_counts = np.array(result_neg_counts)

    validation_counts = (valid_pos_counts, valid_neg_counts) #pack

    assert(strong_hashes.shape[0] == len(valid_pos_counts)) #each hash should have a corresponding pos_count and neg_count from the validation set
    assert(strong_hashes.shape[0] == len(valid_neg_counts))

    return validation_counts


#get the best ratio of positive to total, and the overall percentage
def extract_data(counts):
    pos_counts, neg_counts = counts #unpack

    #calculate percentages avoiding divide by 0
    percentages = np.divide(
        pos_counts,
        pos_counts + neg_counts,
        out=np.zeros_like(pos_counts, dtype=float),
        where=(pos_counts + neg_counts) != 0
    )

    #best percentage
    best_percentage = np.max(percentages)
    best_percentage_idx = np.argmax(percentages)

    best_pos_count = pos_counts[best_percentage_idx]
    best_neg_count = neg_counts[best_percentage_idx]

    best_ratio = (best_pos_count, best_neg_count) #pack

    total_pos_count = np.sum(pos_counts)
    total_neg_count = np.sum(neg_counts)

    total_ratio = (total_pos_count, total_neg_count)

    #returns float, (int, int), (int, int)
    return best_percentage, best_ratio, total_ratio

        


RADIUS = [2]
PN = [512] #max features
KC = [i for i in range(100, 3001, 100)] #hdv dimensions 
DENSITY = [5]
K = [1]
BINARY_HASHING = [False]

TARGET = "MAPK1"
SEED = 42
MIN_STRONG_HASHES = 5
OUTPUT_FILE_NAME = "RESULTS2.csv"

parameters_columns = ['Radius',     #ecfp radius
                      'PN',         #ecfp bit length 
                      'KC',         #projection dimension 
                      'Density',    #connections per KC 
                      'K',          #hash length 
                      'Binary']     #hashing method
output_columns = ['Total buckets', 
                  'Singletons',                     #number of buckets with only one hash 
                  'Biggest bucket',                 #highest number of molecules to one hash 
                  'Median bucket size',             #average bucket size
                  'Strong hashes',                  #number of good hashes selected from the training set, >=0.5 likelihood positive class to validate on, minimum 5 
                  'Best training percentage',       #positive molecules / total molecules from training set 
                  'Overall training fraction',      #positive molecules / total molecules
                  'Best validation percentage',     #from the best hashes, the best validation result 
                  'Overall validation fraction',    #result from strong hashes
                  'Time',                           #time of this cycle
                  ]    

combined_columns = parameters_columns + output_columns

df = pd.DataFrame(columns=combined_columns)

#collect all combinations of parameters
parameter_combinations = list(itertools.product(BINARY_HASHING, K, RADIUS, PN, KC, DENSITY))

for i, (binary_hashing, k, radius, pn, kc, density) in enumerate(parameter_combinations):
    print(f'i: {i}\tRADIUS: {radius}\tPN: {pn}\t\tKC: {kc}\t\tDENSITY: {density}\tK: {k}\tBINARY_HASHING: {binary_hashing}')

    pos_train_features = np.load(f'saved_hashing/{TARGET}_feat{pn}_rad{radius}_active_T.npy')
    neg_train_features = np.load(f'saved_hashing/{TARGET}_feat{pn}_rad{radius}_inactive_T.npy')

    pos_valid_features = np.load(f'saved_hashing/{TARGET}_feat{pn}_rad{radius}_active_V.npy')
    neg_valid_features = np.load(f'saved_hashing/{TARGET}_feat{pn}_rad{radius}_inactive_V.npy')

    start_time = time.time()

    #get a projection with the same seed
    projection = make_projection_matrix(pn, kc, density, seed=SEED)

    #hash the training set positive class and the negative class using the same projection and parameter values
    pos_train_data = hash(pos_train_features, projection, k, binary_hashing)
    neg_train_data = hash(neg_train_features, projection, k, binary_hashing)

    #combine the training hashes
    train_hashes, train_counts = combine_hashes(pos_train_data, neg_train_data, verbose=False)

    #run analysis to extract the strongest hashes, and some other information
    train_strong_hashes, train_strong_counts, train_results = train_analysis(train_hashes, train_counts, min_strong_hashes=MIN_STRONG_HASHES, verbose=False)
    train_best_percentage, train_best_ratio, train_total_ratio = extract_data(train_strong_counts)

    #hash the validation set
    pos_valid_data = hash(pos_valid_features, projection, k, binary_hashing)
    neg_valid_data = hash(neg_valid_features, projection, k, binary_hashing)

    #results of validation
    valid_counts = validate(train_strong_hashes, pos_valid_data, neg_valid_data, verbose=False)
    valid_best_percentage, valid_best_ratio, valid_total_ratio = extract_data(valid_counts)

    end_time = time.time()
    elapsed_time = end_time - start_time

    #collect the data to put into the row of the dataframe
    row_data_list = [
        radius,
        pn,
        kc,
        density,
        k,
        binary_hashing,     
        train_results[0],   #total buckets
        train_results[1],   #singletons
        train_results[2],   #largest bucket size
        train_results[3],   #median bucket size
        train_results[4],   #number of strong hashes
        train_best_percentage,
        f'\'{train_total_ratio[0]}/{train_total_ratio[0] + train_total_ratio[1]}',    #overall train fraction
        valid_best_percentage,
        f'\'{valid_total_ratio[0]}/{valid_total_ratio[0] + valid_total_ratio[1]}',    #overall valid fraction
        elapsed_time,                                                                 #time
    ]

    assert(len(combined_columns) == len(row_data_list))

    df.loc[len(df)] = row_data_list

    


df.to_csv(OUTPUT_FILE_NAME, index=False)


