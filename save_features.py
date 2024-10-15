import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pandas as pd
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

HDV_DIM = 1000
MAX_FEATURES = 512

RADIUS = 2
TARGET="MAPK1"

DATASET = "V"

posFileName = f"datasets/unbiased/{TARGET}/active_{DATASET}.smi"
negFileName = f"datasets/unbiased/{TARGET}/inactive_{DATASET}.smi"

#molecule data from files
pos_df = algo.get_dataframe(posFileName, actual=1)
neg_df = algo.get_dataframe(negFileName, actual=0)

#features of each molecule
pos_features = algo.get_features_matrix(pos_df, RADIUS, MAX_FEATURES)
neg_features = algo.get_features_matrix(neg_df, RADIUS, MAX_FEATURES)

np.save(f'saved_hashing/{TARGET}_feat{MAX_FEATURES}_rad{RADIUS}_active_{DATASET}.npy', pos_features)
np.save(f'saved_hashing/{TARGET}_feat{MAX_FEATURES}_rad{RADIUS}_inactive_{DATASET}.npy', neg_features)