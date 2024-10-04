import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pandas as pd
import HD_library as hd
from sklearn.model_selection import train_test_split

DIM = 40
NBITS = 80

np.random.seed(42)

def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=NBITS)
    npfp = np.zeros((1,), dtype=np.int32)
    DataStructs.ConvertToNumpyArray(fp, npfp)
    return npfp


#build random projection matrix, where each 1 (presence of substructure) corresponds to a 10000 element hdv
feature_matrix = np.random.choice([1, -1], size=(NBITS, DIM))
print(np.hstack((np.arange(feature_matrix.shape[0])[:, np.newaxis], feature_matrix)))




#use fingerprints as a mask to get the binded hdv for a molecule
def get_encoding(features):
    mask = features.astype(bool)
    selected_rows = feature_matrix[mask, :]
    binded = np.prod(selected_rows, axis=0)
    return binded

#fingerprint and then encode using projection and binding
#shape of hdvs = DIM
def encode(df):
    df['features'] = df['Text'].apply(get_fingerprint)
    df['encoding'] = df['features'].apply(get_encoding)
    return df

TARGET="MAPK1"
file_pos = f"datasets/unbiased/{TARGET}/active_T.smi"
file_neg = f"datasets/unbiased/{TARGET}/inactive_T_small.smi"


np.set_printoptions(threshold=np.inf, linewidth=1000)






#keep track of fingerprint/feature usage
column_titles = ['feature number', 'AM+ use count', 'AM+ encoding +1 bit count', 'AM+ encoding -1 bit count', 'AM- use count', 'AM- encoding +1 bit count', 'AM- encoding -1 bit count']
feature_df = pd.DataFrame(columns=column_titles, index=range(NBITS))
feature_df['feature number'] = feature_df.index

#initialize empty numpy arrays to accumulate in
init_columns = ['AM+ encoding +1 bit count', 'AM+ encoding -1 bit count', 'AM- encoding +1 bit count', 'AM- encoding -1 bit count']
for col in init_columns:
    feature_df[col] = [np.zeros(DIM, dtype=np.int32) for _ in range(NBITS)] #how many times a molecule that is a positive case encodes +1 in each bit position

#tracks how each feature affects each bit in the encoding process
def track_usage(df):
    for index, row in df.iterrows(): #each molecule
        AM = 'AM-'
        if (row['actual'] == 1):
            AM = 'AM+'
        
        #extract encodings and which bits affected the overall AMs
        encoding = row['encoding']

        plus_contribution = np.where(encoding == 1, 1, 0)
        minus_contribution = np.where(encoding == -1, 1, 0)

        #extract features, then keep track of how many encodings were affected by the bits of the feature
        features = row['features']
        feature_numbers = np.where(features == 1)[0]
        for feature_number in feature_numbers:
            feature_df.at[feature_number, f'{AM} encoding +1 bit count'] += plus_contribution
            feature_df.at[feature_number, f'{AM} encoding -1 bit count'] += minus_contribution





#fingerprint and encode positive class
pos = hd.read_to_df(file_pos)
pos['actual'] = 1
pos = encode(pos)
print(pos)

pos_features_sum = np.sum(pos['features'].tolist(), axis=0) #how many times each feature is used for AM+
feature_df['AM+ use count'] = pos_features_sum

track_usage(pos)


#fingerprint and encode negative class
neg = hd.read_to_df(file_neg)
neg['actual'] = 0
neg = encode(neg)

neg_features_sum = np.sum(neg['features'].tolist(), axis=0) #how many times each feature is used for AM-
feature_df['AM- use count'] = neg_features_sum

track_usage(neg)






print(feature_df)






#encoded associative memories
encoded_pos = pos['encoding'].tolist()
am_pos = np.sum(encoded_pos, axis=0)
print('am_pos ', am_pos)


encoded_neg = neg['encoding'].tolist()
am_neg = np.sum(encoded_neg, axis=0)
print('am_neg ', am_neg)






#make predictions initially
def check_closer_to(x, am1, am2):
    return 1 if hd.cos_sim(x, am1) > hd.cos_sim(x, am2) else 0

test = pd.concat([pos, neg], axis=0)
test["predicted"] = test["encoding"].apply(lambda vec: check_closer_to(vec, am_pos, am_neg))

hd.print_results(test, "actual", "predicted")







#LETS LOOK AT MOLECULES 0 AND 1
i = 0
print('\nmolecule index: ', i)

dotwithpos = np.dot(encoded_pos[i], am_pos)
dotwithneg = np.dot(encoded_pos[i], am_neg)

print('dot products with am_pos and am_neg: ', dotwithpos, dotwithneg)




#MERGE AMs
def which_bits_to_flip(x, y, a, b):
    xy = np.sign(x * y)
    xy_min_dist = np.min([np.abs(x), np.abs(y)], axis=0)

    xayb = np.sign((x + a) * (y + b))
    xayb_min_dist = np.min([np.abs(x + a), np.abs(y + b)], axis=0)

    conditions = [
        (xy < xayb), #only xy polar -> False
        (xy > xayb), #only xyab polar -> True
        (xy == -1) & (xy_min_dist < xayb_min_dist), #both polar and xyab is further from 0 -> True
        (xy == 1) & (xy_min_dist > xayb_min_dist) #both nonpolar and xyab is closer to 0 -> True
    ]
    choices = [
        1, #False,
        -1, #True,
        -1, #True,
        -1, #True
    ]

    mask = np.select(conditions, choices, default=1)
    
    return mask





j = 1
print('feature ', j)

diff1 = (feature_df.at[j, 'AM+ encoding +1 bit count'] - feature_df.at[j, 'AM+ encoding -1 bit count']) * -2
diff2 = (feature_df.at[j, 'AM- encoding +1 bit count'] - feature_df.at[j, 'AM- encoding -1 bit count']) * -2

mask = which_bits_to_flip(am_pos, am_neg, diff1, diff2)

print(np.vstack((feature_matrix[j, :], mask)))




print()
print(np.vstack((am_pos, am_neg, diff1, diff2)))
print() 

#FLIPBIT
feature_matrix[j, :] *= mask
pos = encode(pos)
neg = encode(neg)
  
#encoded associative memories
encoded_pos = pos['encoding'].tolist()
am_pos = np.sum(encoded_pos, axis=0)
print('am_pos ', am_pos)


encoded_neg = neg['encoding'].tolist()
am_neg = np.sum(encoded_neg, axis=0)
print('am_neg ', am_neg)

dotwithpos = np.dot(encoded_pos[i], am_pos)
dotwithneg = np.dot(encoded_pos[i], am_neg)

print('dot products with am_pos and am_neg: ', dotwithpos, dotwithneg)





#make more predictions
test = pd.concat([pos, neg], axis=0)
test["predicted"] = test["encoding"].apply(lambda vec: check_closer_to(vec, am_pos, am_neg))

hd.print_results(test, "actual", "predicted")









'''for i in range(len(encoded_pos)):
    print('\nmolecule index: ', i)

    dotwithpos = np.dot(encoded_pos[i], am_pos)
    dotwithneg = np.dot(encoded_pos[i], am_neg)

    print('dot products with am_pos and am_neg: ', dotwithpos, dotwithneg)'''





'''fp_pos = pos['FP'].tolist()
fp_pos_sum = np.sum(fp_pos, axis=0)


fp_neg = neg['FP'].tolist()
fp_neg_sum = np.sum(fp_neg, axis=0)

fp_all = fp_pos_sum + fp_neg_sum

print('indices and sums of fp_pos, fp_neg, and fp_all:')
print(np.vstack((np.arange(len(fp_all)), fp_pos_sum, fp_neg_sum, fp_all)))



for i in range(len(encoded_pos)):
    print('\nmolecule index: ', i)

    dotwithpos = np.dot(encoded_pos[i], am_pos)
    dotwithneg = np.dot(encoded_pos[i], am_neg)

    print('dot products with am_pos and am_neg: ', dotwithpos, dotwithneg)

    if (not (dotwithpos > 0 and dotwithneg < 0)):
        print('FAIL')
        
        #print
        print('molecule fingerprint/features')
        print(fp_pos[i])

        #show the frequencies of the fingerprints that are active in this molecule
        print('\nindices, molecule fingerprint masked over all fingerprint frequency')
        print(np.vstack((np.arange(len(fp_all)), np.where(fp_pos[i].astype(bool), fp_all, 0))))


        #visualize encoding and AMs
        print('\nindices, molecule encoding, am_pos, am_neg')
        print(np.vstack((np.arange(len(am_pos)), encoded_pos[i], am_pos, am_neg)))

        #see where the signs are the same, since this is causing less difference in dot product
        same_sign_indices = np.where(np.sign(am_pos) == np.sign(am_neg))[0]
        print('\nindices where signs of am_pos and am_neg are the same')
        print(same_sign_indices)

        def flip_bit_in_vector(feature_number, index_in_feature):
            print('\nflipping feature ', feature_number, ' bit ', index_in_feature)
            feature_matrix[feature_number, index_in_feature] *= -1
            #recalculate AMs
            pos['ReEncoded'] = pos['FP'].apply(get_encoding)  
            neg['ReEncoded'] = neg['FP'].apply(get_encoding)

            encoded_pos = pos['ReEncoded'].tolist()
            am_pos = np.sum(encoded_pos, axis=0)

            encoded_neg = neg['ReEncoded'].tolist()
            am_neg = np.sum(encoded_neg, axis=0)

            #revisualize encoding and AMs
            print('\nindices, molecule encoding, am_pos, am_neg')
            print(np.vstack((np.arange(len(am_pos)), encoded_pos[i], am_pos, am_neg)))

            #recalculate dot products
            dotwithpos = np.dot(encoded_pos[i], am_pos)
            dotwithneg = np.dot(encoded_pos[i], am_neg)
            print('dot products with am_pos and am_neg: ', dotwithpos, dotwithneg)

        flip_bit_in_vector(1, 0)

        print('dot products with am_pos and am_neg: ', dotwithpos, dotwithneg)
        #select the first feature that is used the least overall, and then flip the bit in that feature vector
        feature_number = 44#np.argmin(np.where(fp_pos[i].astype(bool), fp_all, np.inf))
        print('\nfeature number to be modified, number of times used')
        print(feature_number, fp_all[feature_number])

        #show that feature vector
        print('\nfeature vector')
        print(feature_matrix[feature_number, :])
        print(feature_matrix[feature_number, :].shape)

        #FLIP THE BIT IN THE FEATURE VECTOR
        idx = 0#same_sign_indices[0]
        feature_matrix[feature_number, idx] *= -1
        print('\nfeature vector after FLIPPING bit at index: ', idx)
        print(feature_matrix[feature_number, :])

        #recalculate AMs
        pos['ReEncoded'] = pos['FP'].apply(get_encoding)  
        neg['ReEncoded'] = neg['FP'].apply(get_encoding)

        encoded_pos = pos['ReEncoded'].tolist()
        am_pos = np.sum(encoded_pos, axis=0)

        encoded_neg = neg['ReEncoded'].tolist()
        am_neg = np.sum(encoded_neg, axis=0)

        #revisualize encoding and AMs
        print('\nindices, molecule encoding, am_pos, am_neg')
        print(np.vstack((np.arange(len(am_pos)), encoded_pos[i], am_pos, am_neg)))

        #recalculate dot products
        dotwithpos = np.dot(encoded_pos[i], am_pos)
        dotwithneg = np.dot(encoded_pos[i], am_neg)

        print('dot products with am_pos and am_neg: ', dotwithpos, dotwithneg)


        break


'''