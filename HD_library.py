import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

#DIM = 1000
#GRAM_LEN = 3
#SEED = 42

# make a new hyperdimensional vector
def HDV(D):
    values = [-1, 1]
    return np.random.choice(values, size=(1, D))

# cosine similarity between two vectors
def cos_sim(x, y):
    num = (x @ y.T).item()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom

def bundle(hdvs):
    bundled = np.sum(hdvs, axis=0)
    bundled = np.where(bundled > 0, 1, np.where(bundled < 0, -1, 0))
    return bundled

def bind(hdvs):
    return np.prod(hdvs, axis=0)

def shift(x, step):
    return np.roll(x, step)

def sign(hdv):
    return np.where(hdv > 0, 1, np.where(hdv < 0, -1, 0))


# read to a dataframe
def read_to_df(filename):
    df = pd.read_csv(filename, header=None, names=['Text'])

    getID = lambda s: s.split(' ', 1)[1]

    df['ID'] = df['Text'].apply(getID)

    # remove id number at end of row
    getText = lambda s: s.split(' ', 1)[0]
    
    df['Text'] = df['Text'].apply(getText)
    return df

# extract unique chars
def get_unique_chars(df):
    text = df['Text']
    corpus = ''.join(text)
    unique = set(corpus)

    return unique

# bind together a molecule and add into a column of a dataframe
def bind_ngrams(df, ngram_dict, n, D):    
    def bind_together(entry): 
        num_grams = len(entry) - n + 1

        # collect all ngrams
        hdvs = np.zeros((num_grams, D))
        for i in range(num_grams):
            ngram = entry[i : i + n]
            hdvs[i] = ngram_dict[ngram]
        hdv = bind(hdvs)
        return hdv
    
    tqdm.pandas(desc="Encoding texts", unit="molecule")

    df['Encoded'] = df['Text'].progress_apply(bind_together)
    return


'''# bundle together a molecule and add into a column of a dataframe
def bundle_ngrams(df, ngram_dict, n, D):    
    def bundle_together(entry): 
        num_grams = len(entry) - n + 1

        # collect all ngrams
        hdvs = np.zeros((num_grams, D))
        for i in range(num_grams):
            ngram = entry[i : i + n]
            hdvs[i] = ngram_dict[ngram]
        hdv = bundle(hdvs)
        return hdv
    
    tqdm.pandas(desc="Bundling texts", unit="molecule")

    df['Bundled'] = df['Text'].progress_apply(bundle_together)
    return
'''

'''# bundle together a molecule and add to a column of df
def bundle_ngrams_multi(df, dict1, dict2, D):

    # unpacks tuples of dictionary and n
    ngram_dict1, n1 = dict1
    ngram_dict2, n2 = dict2

    def bundle_together(entry): 
        # collect all ngrams for dict1
        num_grams = len(entry) - n1 + 1
        hdvs1 = np.zeros((num_grams, D))
        for i in range(num_grams):
            ngram = entry[i : i + n1]
            hdvs1[i] = ngram_dict1[ngram]

        # collect all ngrams for dict2
        num_grams = len(entry) - n2 + 1
        hdvs2 = np.zeros((num_grams, D))
        for i in range(num_grams):
            ngram = entry[i : i + n2]
            hdvs2[i] = ngram_dict2[ngram]

        allhdvs = np.vstack((hdvs1, hdvs2))
    
        hdv = bundle(allhdvs)
        return hdv
    
    tqdm.pandas(desc="Bundling texts")

    df['Bundled'] = df['Text'].progress_apply(bundle_together)
    return
'''

# given a dictionary make an n-gram dictionary
def make_dict(vecs, n):
    tups = itertools.product(vecs.keys(), repeat=n)

    gram_name = lambda tup: ''.join(tup)

    # yields the rotated and binded hdv
    def gram_vec(tup):

        all_vecs = vecs[tup[0]]

        for i in range(1, len(tup)):
            shifted_vec = shift(vecs[tup[i]], i)
            all_vecs = np.vstack((all_vecs, shifted_vec))
        
        return bind(all_vecs)
    
    # make dictionary with n-grams
    gram_dict = {gram_name(tup): gram_vec(tup) for tup in tqdm(tups, desc="Building dictionary", unit="n-gram")}

    #print(len(gram_dict.keys()))
    #print(vecs.keys())
    return gram_dict


def bundle_clusters_dbscan(hdvs):
    pca = PCA(n_components=10)
    reduced = pca.fit_transform(np.vstack(hdvs))

    db = DBSCAN(eps=0.5, min_samples=1)
    labels = db.fit_predict(reduced)

    clusters = {}
    for label in set(labels):
        if label != -1:
            clusters[label] = [hdvs[i] for i in range(len(hdvs)) if labels[i] == label]

    print(len(clusters.keys()))
    
    bundled_clusters = []
    for cluster in clusters.values():
        bundled_cluster = bundle(np.vstack(cluster))
        bundled_clusters.append(bundled_cluster)
    
    return bundled_clusters




#using a training df and the predictions retrains am
def retrain(df, actual, predicted, positive_am, negative_am):
    false_negatives = df[(df[actual] == 1) & (df[predicted] == 0)]
    false_positives = df[(df[actual] == 0) & (df[predicted] == 1)]

    for e in false_negatives["Encoded"].tolist():
        positive_am += e
        negative_am -= e

    for e in false_positives["Encoded"].tolist():
        positive_am -= e
        negative_am += e

    return positive_am, negative_am



def print_results(df, actual, predicted):
    active_rows = df[df[actual] == 1]
    active_correct = (active_rows[actual] == active_rows[predicted]).sum()
    active_total = active_rows.shape[0]

    print(f"Actives accuracy: {active_correct/active_total} ({active_correct}/{active_total})")

    inactive_rows = df[df[actual] == 0]
    inactive_correct = (inactive_rows[actual] == inactive_rows[predicted]).sum()
    inactive_total = inactive_rows.shape[0]

    print(f"Inactives accuracy: {inactive_correct/inactive_total} ({inactive_correct}/{inactive_total})")
    return



'''# bundle together based on a threshold of cosine similarity using BFS
def bundle_clusters(hdvs, threshold=0.5):

    #DEBUG
    print(f"THRESHOLD {threshold}")

    n = len(hdvs)
    unvisited = set(range(n))

    clusters = []

    # progress bar
    with tqdm(total=n, desc="Making clusters", unit="HDV") as pbar:

        # new cluster
        while len(unvisited) > 0:
            cluster = []
            queue = [unvisited.pop()]

            # find neighbors
            while len(queue) > 0:
                row = queue.pop(0)
                cluster.append(hdvs[row])

                pbar.update(1)

                # check all remaining vectors, remove from remaining set, and add to search queue
                neighbors = []
                for neighbor in unvisited:  
                    if cos_sim(hdvs[neighbor], hdvs[row]) > threshold:

                        # limit size of cluster
                        """include = True
                        for element in cluster:
                            if (cos_sim(hdvs[neighbor], element) < threshold):
                                include = False
                        
                        if include:"""
                        neighbors.append(neighbor)

                # avoid set change size during iteration error 
                for neighbor in neighbors:
                    unvisited.remove(neighbor)
                    queue.append(neighbor)
            
            # done with this cluster
            clusters.append(cluster)
            #pbar.update(len(cluster))

    #DEBUG
    for i in range(len(clusters)):
        if (len(clusters[i]) > 1):
            print(f"cluster {i}, number of elements: {len(clusters[i])}")

    # bundle together each cluster
    bundled_clusters = []
    for cluster in clusters:
        bundled_cluster = bundle(np.vstack(cluster))
        bundled_clusters.append(bundled_cluster)
    
    return bundled_clusters'''