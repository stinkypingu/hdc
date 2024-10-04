import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pandas as pd
import HD_library as hd
from sklearn.model_selection import train_test_split

ROWS = 3
DIM = 10000

x = hd.HDV(DIM)
y = hd.HDV(DIM)
z = hd.HDV(DIM)
a = hd.HDV(DIM)
b = hd.HDV(DIM)
c = hd.HDV(DIM)


xy = x * y
xyz = x * y * z
xyzabc = x * y * z * a * b * c

#print(np.sum(x == xy))
#print(np.sum(x == xyz))
#print(np.sum(x == xyzabc))
print(np.sum((x == xy) & (x == xyz)))

for i in [xy]:
    print(np.sum(i == xyz))