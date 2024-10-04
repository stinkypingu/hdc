import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pandas as pd
import HD_library as hd

def getFingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

    npfp = np.zeros((1,), dtype=int)

    DataStructs.ConvertToNumpyArray(fp, npfp)

    return npfp

TARGET="MAPK1"
actives_file_name = f"datasets/unbiased/{TARGET}/active_T.smi"

df = hd.read_to_df(actives_file_name)
df['count'] = df['Text'].apply(getFingerprint)

print(df)
#print(df['count'].mean())
#print(df['count'].max())


print(len(df))
# Compare each pair of arrays
for i in range(len(df)):
    for j in range(i, len(df)):

        if (df.loc[i, 'Text'] == df.loc[j, 'Text'] and i != j):
            print(f"rows {i}, {j}. ids: {df.loc[i, 'ID']}, {df.loc[j, 'ID']}")

#        if (np.array_equal(df.loc[i, 'count'], df.loc[j, 'count']) and i != j):
#            print(i, j)
            
a = getFingerprint("COc1ccc(cc1OC)c2nn(C)c3sc(cc23)C(=O)NCC[NH+](C)C") #24398679
b = getFingerprint("COc1ccc(cc1OC)c2nn(C)c3sc(cc23)C(=O)NCC[NH+](C)C") #49667886

print(np.sum(a-b))



datasets = ["FEN1", "ADRB2", "ALDH1", "ESR1_ago", "ESR1_ant", "GBA", "IDH1", "KAT2A", "MAPK1", "MTORC1", "OPRK1", "PKM2", "PPARG", "TP53", "VDR"]

for dataset in datasets:
    filename = f"datasets/unbiased/{dataset}/active_T.smi"
    df = hd.read_to_df(filename)

    print(f"\n{dataset}")
    
    dups = 0
    for i in range(len(df)):
        for j in range(i+1, len(df)):

            if (df.loc[i, 'Text'] == df.loc[j, 'Text'] and df.loc[i, 'ID'] != df.loc[j, 'ID']):
                print(f"rows {i}, {j}. ids: {df.loc[i, 'ID']}, {df.loc[j, 'ID']}")
                dups += 1
    
    print(f"dups: {dups}. total: {len(df)}")
