import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pandas as pd
import HD_library as hd
from sklearn.model_selection import train_test_split

DIM = 10000
NBITS = 2048

def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=NBITS)
    npfp = np.zeros((1,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, npfp)
    return npfp


#build random projection matrix, where each 1 (presence of substructure) corresponds to a 10000 element hdv
projection_matrix = np.random.choice([1, -1], size=(NBITS, DIM))

#use fingerprints as a mask to get the binded hdv for a molecule
def mask_bind(fp):
    mask = fp.astype(bool)
    selected_rows = projection_matrix[mask, :]
    binded = np.prod(selected_rows, axis=0)
    return binded

#fingerprint and then encode using projection and binding
#shape of hdvs = DIM
def encode(df):
    df['FP'] = df['Text'].apply(get_fingerprint)
    df['Encoded'] = df['FP'].apply(mask_bind)
    return df

#fingerprinting only
#shape of hdvs = NBITS
def encode2(df):
    df['FP'] = df['Text'].apply(get_fingerprint)
    df['Encoded'] = df['FP'].apply(lambda x: np.where(x == 0, 1, -1)) #more 0s than 1s, so map the 1s to -1s
    return df




TARGET="MAPK1"
actives_file_name = f"datasets/unbiased/{TARGET}/active_T.smi"
inactives_file_name = f"datasets/unbiased/{TARGET}/inactive_T_large.smi"

#fingerprint and encode
actives = hd.read_to_df(actives_file_name)
actives = encode2(actives)

inactives = hd.read_to_df(inactives_file_name)
inactives = encode2(inactives)




'''#DEBUG
actives['a'] = actives['FP'].apply(lambda vec: np.where(vec == 1)[0]) #indices where bit is 1
indices = np.hstack(actives['a'].tolist())
values, counts = np.unique(indices, return_counts=True)
for i in range(len(values)):
    print(values[i], counts[i])

inactives['a'] = inactives['FP'].apply(lambda vec: np.where(vec == 1)[0])
indices = np.hstack(inactives['a'].tolist())
count = np.count_nonzero(indices == 1750)
print(count)
#print(inactives['count'].tolist())'''

np.set_printoptions(precision=4)


#set true answers
actives["Actual"] = 1
inactives["Actual"] = 0

#split
train_actives, test_actives = train_test_split(actives, test_size=0.2, random_state=42)
train_inactives, test_inactives = train_test_split(inactives, test_size=0.2, random_state=42)

train = pd.concat([train_actives, train_inactives], axis=0)
test = pd.concat([test_actives, test_inactives], axis=0)

#get associative memories
def build_memory(df):
    encodings = np.vstack(df['Encoded'].tolist())
    bundled = np.sum(encodings, axis=0)
    return bundled

actives_am = build_memory(train_actives)
inactives_am = build_memory(train_inactives)

#actives_am = hd.sign(actives_am)
#inactives_am = hd.sign(inactives_am)

print(actives_am.shape)


#DEBUG am
print(f"actives am: {actives_am}")
print(f"inactives am: {inactives_am}")


'''#DEBUG dim reduce
def min_max_norm(vec):
    min = np.min(vec)
    max = np.max(vec)

    if (max - min == 0):
        return vec
    return 2 * (vec - min) / (max - min) - 1

actives_am = min_max_norm(actives_am)
inactives_am = min_max_norm(inactives_am)

print(f"actives am: {actives_am}")
print(f"inactives am: {inactives_am}")

difference = actives_am - inactives_am
dropmask = np.abs(difference) > 0.8
print(dropmask)
print(np.sum(dropmask))
actives_am = actives_am[dropmask]
inactives_am = inactives_am[dropmask]

test['Encoded'] = test['Encoded'].apply(lambda vec: vec[dropmask])'''



 
#predictions
def check_closer_to(am, x, y):
    return 1 if hd.cos_sim(am, x) > hd.cos_sim(am, y) else 0

test["Predicted"] = test["Encoded"].apply(lambda vec: check_closer_to(vec, actives_am, inactives_am))

test["active_closeness"] = test["Encoded"].apply(lambda vec: hd.cos_sim(vec, actives_am))
test["inactive_closeness"] = test["Encoded"].apply(lambda vec: hd.cos_sim(vec, inactives_am))

print(test[["active_closeness", "inactive_closeness", "Actual", "Predicted"]])

#printing
hd.print_results(test, "Actual", "Predicted")



for i in range(10):
    print(f"epoch {i+1}")


    train["Predicted"] = train["Encoded"].apply(lambda vec: check_closer_to(vec, actives_am, inactives_am))
    actives_am, inactives_am = hd.retrain(train, "Actual", "Predicted", actives_am, inactives_am)

    #actives_am = hd.sign(actives_am)
    #inactives_am = hd.sign(inactives_am)


    #DEBUG am
    print(f"actives am: {actives_am}")
    print(f"inactives am: {inactives_am}")


    print("TRAINING SET")
    hd.print_results(train, "Actual", "Predicted")


    test["Predicted"] = test["Encoded"].apply(lambda vec: check_closer_to(vec, actives_am, inactives_am))

    print("TEST SET")
    hd.print_results(test, "Actual", "Predicted")


    #DEBUG closeness
    test["active_closeness"] = test["Encoded"].apply(lambda vec: hd.cos_sim(vec, actives_am))
    test["inactive_closeness"] = test["Encoded"].apply(lambda vec: hd.cos_sim(vec, inactives_am))
    print(test[["active_closeness", "inactive_closeness", "Actual", "Predicted"]])

    print()




'''
arrs = np.vstack(df['FP'].tolist())
print(arrs.shape)
sums = np.sum(arrs, axis=0)
#np.set_printoptions(threshold=np.inf)
print(sums)
print(sums.max())
print(np.count_nonzero(sums == 0))

projection_matrix = np.random.choice([1, -1], size=(1024, 10000))
print(projection_matrix)
print(projection_matrix.shape)


print(np.sum(arrs[0, :])) #first fingerprint
mask = arrs[0, :].astype(bool)
print(mask)

selected = projection_matrix[mask, :]
print(selected.shape)

res = np.prod(selected, axis=0)
print(res)
print(res.shape)
'''