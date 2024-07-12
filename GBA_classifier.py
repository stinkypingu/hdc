import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import HD_library as hd
import pandas as pd
from sklearn.model_selection import train_test_split

# read to dataframe
df_actives = hd.read_to_df("GBA_actives.txt")
df_inactives = hd.read_to_df("GBA_inactives.txt")

# extract unique chars
actives_unique_chars = hd.get_unique_chars(df_actives)
inactives_unique_chars = hd.get_unique_chars(df_inactives)

unique_chars = actives_unique_chars | inactives_unique_chars

# make dictionary to HDVs
char_dict = {char: hd.HDV(1000) for char in unique_chars}
print(f"unique character count: {len(char_dict)}")

bigram_dict = hd.make_dict(char_dict, 3)
print(f"bigram count: {len(bigram_dict)}\n")


# bundle together each entry
hd.bundle_ngrams(df_actives, bigram_dict, 3)
#print(df_actives)

hd.bundle_ngrams(df_inactives, bigram_dict, 3)
#print(df_inactives)

# bundle together random 80% of training data to create profiles
train_actives, test_actives = train_test_split(df_actives, test_size=0.2, random_state=42)
actives_profile_hdv = hd.bundle(np.vstack(train_actives["Bundled"].tolist()))

train_inactives, test_inactives = train_test_split(df_inactives, test_size=0.2, random_state=42)
inactives_profile_hdv = hd.bundle(np.vstack(train_inactives["Bundled"].tolist()))

# TESTING
test_actives["Actual"] = 1
test_inactives["Actual"] = 0

def check_closer_to(hdv, a, b):
    return 1 if hd.cos_sim(hdv, a) > hd.cos_sim(hdv, b) else 0

# results for test actives data
test_actives["Predicted"] = test_actives["Bundled"].apply(lambda vec: check_closer_to(vec, actives_profile_hdv, inactives_profile_hdv))
print(test_actives)

active_acc = (test_actives["Actual"] == test_actives["Predicted"]).mean()
print(f"Actives accuracy: {active_acc}")


# results for test inactives data
test_inactives["Predicted"] = test_inactives["Bundled"].apply(lambda vec: check_closer_to(vec, actives_profile_hdv, inactives_profile_hdv))
print(test_inactives)

inactive_acc = (test_inactives["Actual"] == test_inactives["Predicted"]).mean()
print(f"Inactives accuracy: {inactive_acc}")







#total = np.zeros((1, 1000))
#for c in entry:
#    total = hd.bind(total, char_dict[c])
    
