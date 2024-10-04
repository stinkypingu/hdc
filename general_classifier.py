import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import HD_library as hd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tqdm import tqdm

DIM = 10000
GRAM_LEN = 2
SEED = 42

TEST_SPLIT = 0.2

TARGET="MAPK1"
actives_file_name = f"datasets/Lit-PCBA/{TARGET}/actives.smi"
inactives_file_name = f"datasets/Lit-PCBA/{TARGET}/inactives_small.smi"

# read to dataframe
df_actives = hd.read_to_df(actives_file_name)
df_inactives = hd.read_to_df(inactives_file_name)

# extract unique chars
actives_unique_chars = hd.get_unique_chars(df_actives)
inactives_unique_chars = hd.get_unique_chars(df_inactives)

unique_chars = actives_unique_chars | inactives_unique_chars

# make dictionary to HDVs
char_dict = {char: hd.HDV(DIM) for char in unique_chars}
print(f"unique character count: {len(char_dict)}")

bigram_dict = hd.make_dict(char_dict, GRAM_LEN)

# bind together each entry
hd.bind_ngrams(df_actives, bigram_dict, GRAM_LEN, DIM)
hd.bind_ngrams(df_inactives, bigram_dict, GRAM_LEN, DIM)

# bundle together random 80% of training data to create profiles
df_actives["Actual"] = 1
df_inactives["Actual"] = 0
train_actives, test_actives = train_test_split(df_actives, test_size=TEST_SPLIT, random_state=SEED)
train_inactives, test_inactives = train_test_split(df_inactives, test_size=TEST_SPLIT, random_state=SEED)
print(f"Train/test/split complete.")

actives_profile_hdv = np.sum(np.vstack(train_actives["Encoded"].tolist()), axis=0)
inactives_profile_hdv = np.sum(np.vstack(train_inactives["Encoded"].tolist()), axis=0)





test = pd.concat([test_actives, test_inactives], axis=0)
train = pd.concat([train_actives, train_inactives], axis=0)

def check_closer_to(hdv, a, b):
    return 1 if hd.cos_sim(hdv, a) > hd.cos_sim(hdv, b) else 0

test["Predicted"] = test["Encoded"].apply(lambda vec: check_closer_to(vec, actives_profile_hdv, inactives_profile_hdv))
train["Predicted"] = train["Encoded"].apply(lambda vec: check_closer_to(vec, actives_profile_hdv, inactives_profile_hdv))
print(test)

#printing
hd.print_results(test, "Actual", "Predicted")

#RETRAIN
actives_profile_hdv, inactives_profile_hdv = hd.retrain(train, "Actual", "Predicted", actives_profile_hdv, inactives_profile_hdv)

print("TRAINING SET")
hd.print_results(train, "Actual", "Predicted")


test["Predicted"] = test["Encoded"].apply(lambda vec: check_closer_to(vec, actives_profile_hdv, inactives_profile_hdv))
print("TEST SET")
hd.print_results(test, "Actual", "Predicted")





'''
#CLUSTERING
#actives_clustered_profiles = hd.bundle_clusters(train_actives["Bundled"].tolist(), 0.7)
actives_clustered_profiles = hd.bundle_clusters_dbscan(train_actives["Bundled"].tolist())
print(f"clustering actives complete.")

#inactives_clustered_profiles = hd.bundle_clusters(train_inactives["Bundled"].tolist(), 0.7)
inactives_clustered_profiles = hd.bundle_clusters_dbscan(train_inactives["Bundled"].tolist())
print(f"clustering inactives complete.")


#TESTING
test_actives["Actual"] = 1
test_inactives["Actual"] = 0

test = pd.concat([test_actives, test_inactives], axis=0)
print(f"assigned expected values to test set.")

def closeness(hdv, cluster_profiles):
    largest_cos_sim = -1
    for cluster_profile in cluster_profiles:
        largest_cos_sim = max(largest_cos_sim, hd.cos_sim(hdv, cluster_profile))
    return largest_cos_sim

#redundant
def check_closeness(hdv, cluster_profiles_1, cluster_profiles_2, threshold=0.6):
    if (closeness(hdv, cluster_prof iles_1) > closeness(hdv, cluster_profiles_2)):
        return 1
    return 0

tqdm.pandas(desc="Applying active closeness", unit="HDV")
test["active_closeness"] = test["Bundled"].progress_apply(lambda vec: closeness(vec, actives_clustered_profiles))

tqdm.pandas(desc="Applying inactive closeness", unit="HDV")
test["inactive_closeness"] = test["Bundled"].progress_apply(lambda vec: closeness(vec, inactives_clustered_profiles))

def compare_closeness(row):
    return 1 if row["active_closeness"] > row["inactive_closeness"] else 0

tqdm.pandas(desc="Making predictions", unit="molecule")
test["Predicted"] = test.progress_apply(compare_closeness, axis=1)
print(f"predictions complete.")

print(test)


#OUTPUT
active_rows = test[test["Actual"] == 1]
active_correct = (active_rows["Actual"] == active_rows["Predicted"]).sum()
active_total = active_rows.shape[0]

print(f"Actives accuracy: {active_correct/active_total} ({active_correct}/{active_total})")

inactive_rows = test[test["Actual"] == 0]
inactive_correct = (inactive_rows["Actual"] == inactive_rows["Predicted"]).sum()
inactive_total = inactive_rows.shape[0]

print(f"Inactives accuracy: {inactive_correct/inactive_total} ({inactive_correct}/{inactive_total})")


print()
#print(active_rows)'''





'''

# TESTING
test_actives["Actual"] = 1
test_inactives["Actual"] = 0

def check_closer_to(hdv, a, b):
    return 1 if hd.cos_sim(hdv, a) > hd.cos_sim(hdv, b) else 0

# results for test actives data
test_actives["active_closeness"] = test_actives["Binded"].apply(lambda vec: hd.cos_sim(vec, actives_profile_hdv))
test_actives["inactive_closeness"] = test_actives["Binded"].apply(lambda vec: hd.cos_sim(vec, inactives_profile_hdv))

test_actives["Predicted"] = test_actives["Binded"].apply(lambda vec: check_closer_to(vec, actives_profile_hdv, inactives_profile_hdv))
print(test_actives)

active_acc = (test_actives["Actual"] == test_actives["Predicted"]).mean()
print(f"Actives accuracy: {active_acc}")


# results for test inactives data
test_inactives["Predicted"] = test_inactives["Binded"].apply(lambda vec: check_closer_to(vec, actives_profile_hdv, inactives_profile_hdv))

inactive_acc = (test_inactives["Actual"] == test_inactives["Predicted"]).mean()
print(f"Inactives accuracy: {inactive_acc}")'''







'''#VISUALIZE
data = np.vstack([
    np.array(train_actives["Bundled"]).tolist(),
    np.array(train_inactives["Bundled"]).tolist(),
])
print(data.shape)
data = pd.DataFrame(data)
print(data.shape)
correlation_matrix = data.corr()

subset_correlation_matrix = correlation_matrix.iloc[:50, :50]
strong_correlations = subset_correlation_matrix[(subset_correlation_matrix >= 0.4) | (subset_correlation_matrix <= -0.4)]

plt.figure(figsize=(10, 8))
sns.heatmap(strong_correlations, cmap='coolwarm', annot=True, fmt=".2f", annot_kws={"size": 7})
plt.show()

'''


'''#PCA
combined_data = np.vstack([
    np.array(train_actives["Bundled"]).tolist(),
    np.array(train_inactives["Bundled"]).tolist(),
    np.array(test_actives["Bundled"]).tolist()
])



pca = PCA(n_components=2)
combined_pca = pca.fit_transform(combined_data)

train_actives_pca = combined_pca[:len(train_actives)]
train_inactives_pca = combined_pca[len(train_actives):len(train_actives) + len(train_inactives)]
test_actives_pca = combined_pca[len(train_actives) + len(train_inactives):-2] 

ms = 10
plt.figure(figsize=(8, 6))
plt.scatter(train_inactives_pca[:, 0], train_inactives_pca[:, 1], c='red', s=ms, label='train inactive')
plt.scatter(train_actives_pca[:, 0], train_actives_pca[:, 1], c='blue', s=ms, label='train active')
plt.scatter(test_actives_pca[:, 0], test_actives_pca[:, 1], c='lightblue', s=ms, label='test active')

plt.xlabel('principle component 1')
plt.ylabel('principle component 2')
plt.title('pca of HDVs')
plt.legend()
plt.show()
'''



'''#CLOSENESS
print("CLOSENESS")
def calculate_closeness(hdv):
    return hd.cos_sim(hdv, inactives_profile_hdv)
train_actives["Closeness"] = train_actives["Bundled"].apply(calculate_closeness)
print(f'actives, mean: {train_actives["Closeness"].mean()}, min: {train_actives["Closeness"].min()}, max: {train_actives["Closeness"].max()}')

train_inactives["Closeness"] = train_inactives["Bundled"].apply(calculate_closeness)
print(f'inactives, mean: {train_inactives["Closeness"].mean()}, min: {train_inactives["Closeness"].min()}, max: {train_inactives["Closeness"].max()}')

test_actives["Closeness"] = test_actives["Bundled"].apply(calculate_closeness)
test_inactives["Closeness"] = test_inactives["Bundled"].apply(calculate_closeness)


#PLOT
plt.figure(figsize=(10, 6))
ms = 4

#x = np.arange(len(train_actives["Closeness"]))
#plt.plot(x, train_actives["Closeness"], marker='o', linestyle='None', color='b', label='actives', markersize=ms)

#x = np.arange(len(train_inactives["Closeness"]))
#plt.plot(x, train_inactives["Closeness"], marker='o', linestyle='None', color='r', label='inactives', markersize=ms)

x = np.arange(len(test_actives["Closeness"]))
plt.plot(x, test_actives["Closeness"], marker='o', linestyle='None', color='lightblue', label='actives', markersize=ms)

x = np.arange(len(test_inactives["Closeness"]))
plt.plot(x, test_inactives["Closeness"], marker='o', linestyle='None', color='lightcoral', label='inactives', markersize=ms)

plt.legend()
plt.show()

print()
'''




#total = np.zeros((1, 1000))
#for c in entry:
#    total = hd.bind(total, char_dict[c])
    
