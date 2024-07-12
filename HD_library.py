import numpy as np
import pandas as pd
import itertools

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

def bind(x, y):
    return x * y

def shift(x, step):
    return np.roll(x, step)


# read to a dataframe
def read_to_df(filename):
    df = pd.read_csv(filename, header=None, names=['Text'])

    # remove id number at end of row
    strip = lambda s: s.split(' ', 1)[0]
    
    df['Text'] = df['Text'].apply(strip)
    return df

# extract unique chars
def get_unique_chars(df):
    text = df['Text']
    corpus = ''.join(text)
    unique = set(corpus)

    return unique

# bundle together a molecule and add into a column of a dataframe
def bundle_ngrams(df, ngram_dict, n, D):    
    def bundle_together(entry): 
        num_grams = len(entry) - n + 1
        hdvs = np.zeros((num_grams, D))
        for i in range(num_grams):
            ngram = entry[i : i + n]
            hdvs[i] = ngram_dict[ngram]
        hdv = bundle(hdvs)
        return hdv
    
    df['Bundled'] = df['Text'].apply(bundle_together)
    return


# given a dictionary make an n-gram dictionary
def make_dict(vecs, n):
    tups = itertools.product(vecs.keys(), repeat=n)

    gram_name = lambda tup: ''.join(tup)

    # yields the rotated and binded hdv
    def gram_vec(tup):
        combined_vec = shift(vecs[tup[0]], 0)
        for i in range(1, len(tup)):
            shifted_vec = shift(vecs[tup[i]], i)
            combined_vec = bind(combined_vec, shifted_vec)
        return combined_vec
    
    # make dictionary with n-grams
    gram_dict = {gram_name(tup): gram_vec(tup) for tup in tups}


    #print(len(gram_dict.keys()))
    #print(vecs.keys())
    return gram_dict
