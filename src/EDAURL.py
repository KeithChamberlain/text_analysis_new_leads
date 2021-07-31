import numpy as np
import pandas as pd
import re
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from MineURL import MineURL, ManyURL
from makedv import get_search
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
from math import isclose
from nltk.corpus import stopwords
from sklearn.decomposition import NMF
from joblib import dump, load

def count_words(vectorizer, corpus):
    '''
    count_words: Counts the words fit by CountVectorizer and returns a 
    Series with the relative frequencies.     
    vectorizer - vectorizer object called just before fitting.
    corpus - the bag of words
    returns - wc2 a Series with the words and freq each word
    '''
    word_list=vectorizer.get_feature_names()
    word_counts = dict()
    for word, count in zip(word_list, corpus.sum(axis=0)):
        word_counts[word] = count
    wc = pd.Series(word_counts)
    ttl_words = wc.sum()
    wc2 = wc/ttl_words*100
    wc2.sort_values(ascending=False, inplace=True)
    wc.sort_values(ascending=False, inplace=True)
    return wc, wc2 

def plot_words(keys, values, title="Word Frequency", path="./img/WordFreq_.jpg"):
    '''
    plot_words takes the names and values from a words/count Series, a custom title,
    a custom path, and plots the word frequencies, saving the image.
    keys: the words
    values: the counts
    title: a title for the plot
    path: a path/filename to save the resulting image to disk

    returns: fig, ax tuple for later adjustments
    '''
    fig, ax = plt.subplots(figsize=(16,5))
    ax.bar(keys, values, align='center');
    plt.xticks(rotation=90);
    # Giving the tilte for the plot
    ax.set_title(title, fontsize=35)
    ax.set_xlabel('Words', fontsize=25)
    ax.set_ylabel('Relative Frequency', fontsize=25)
    fig.savefig(path)
    plt.show()
    return fig, ax

def fit_nmf(X, n_words, init = 'random', max_iter=200):
    '''
    fit_nmf fits an NMF object with a-priori n_words to fit
    X: a bag of words to fit
    n_words: how many max words to fit from
    init: initialization parameter; defaults to 'random'
    max_iter, same as NMF(max_iter), defaults to 200.
    returns nmf.reconstruction_err_, W, H matrices
    '''
    nmf = NMF(n_components=n_words, init = init, max_iter=max_iter)
    nmf.fit_transform(X)
    W = nmf.transform(X)
    H = nmf.components_
    #print(H[:10])
    return nmf.reconstruction_err_, W, H

def plot_score(score, words):
    '''
    plot_score given scores, and related words that were
    scored, plot the results.
    score: usually, nmf.reconstruction_err_ results.
    words: associated words that were scored. The H matrix
    returns fig, ax for additional adjustments
    '''
    fig, ax= plt.subplots(figsize=(16,5))
    ax.plot(words, score)
    plt.show()
    return fig, ax

if __name__ == "__main__":
    max_features_base=500
    max_features_extended=500
    max_features_search=500
    
    string = '''
    Custom Stop Words
    '''
    print(string)
    stopwords_ = set({'com','au','edu','www','http','ftp','net', 'fi', 'uk', 'us', 'org','ca', '00', 
        '127','129','10','adme','bbb','ww2','www2','15','150','168','192','ads','ae','at','br','be',
        'bg','biz','br','crm','crm3','cs8','cse','cy','cz','de','dk','edu','eg','es','fi','fr','gr',
        'id','ie','il','in','ir','lm','ly','lt','maps','me','map','mail','na32','na35','na38','na58',
        'na','na61','na78','mx','mt','my','ng','ni','nl','no','nz','pe','ph','pk','pl','pt','ro','rs',
        'ru','rw','s3','sa','se','sg','si','sk','so','tb','th','tr','tt','tw','tz','ua','uk','web',
        'usa','vn','yp','za','https','kjrh','site','sites','yhs4', 'sales', 'yimg','cn','co','ui',
        'pi','pch','qa','s0', 'na104', 'na27', 'na31', 'na74', 'na8', 'na93', 'mc','lu','lv','kr',
        'kh','gy','hq', 'hr', 'ht', 'hu','hk', 'hn','zw', '152', '155', '172', '193', '23', '232', 
        '24','ws', 'www1', 'www4', 'wxyz','xyz','ib','jm', 'jo', 'jp', 'fm', 'en','dp','du','ec','99',
        '64','gclid','bwc','campaign','ad','fba','utm','fbclid','wcb','tsr','bwe'})
    stopwords2 = set(stopwords.words('english')).union(stopwords_)

    string = '''
    Read Data, Verify Input
    '''
    print(string)
    df=pd.read_csv("../data/cleaned1.csv", compression="gzip")
    print(df.info())    
    df["Page"].fillna(" ", inplace=True)

    string = '''
    Generate URL data from URL Class
    '''
    print(string)
    many = ManyURL(df["Page"])
    urlDict = many.get_records()
    url1Data = pd.DataFrame(urlDict, columns=["base_url","extended_url","search_string"])
    url1Data.fillna(" ", inplace=True)
    print(url1Data.info())
    print("Unique Base URLs", len(url1Data["base_url"].unique()))
    uurl = pd.Series(url1Data["base_url"].unique())
    index = get_search(uurl, search_string="google|Google")
    print(np.sum(index))

    string = '''
    Get Bag Of Words & Word Counts (wc)
    For Extended URL
    '''
    print(string)
    index = df["New_Lead"]=="NewLead"
    index2 = df["New_Lead"]!="NewLead"
    vectorizer_extendedNL = CountVectorizer(analyzer="word", max_features = max_features_base, 
        stop_words=stopwords2, ngram_range=(1,3))
    X_extendedNL = vectorizer_extendedNL.fit_transform(url1Data.loc[index, "extended_url"]).toarray()
    wc1, wf1 = count_words(vectorizer_extendedNL, X_extendedNL)

    vectorizer_extendedNNL = CountVectorizer(analyzer="word", max_features = max_features_base, 
        stop_words=stopwords2, ngram_range=(1,3))
    X_extendedNNL = vectorizer_extendedNNL.fit_transform(url1Data.loc[index2, "extended_url"]).toarray()
    wc2, wf2 = count_words(vectorizer_extendedNNL, X_extendedNNL)
    plot_words(wf1.index[:30], wf1[:30], title="Word Frequency-Landing Page-Extended URL-New Leads", 
        path = "../img/WordFreqLPExtURLNL.jpg")
    plot_words(wf2.index[:30], wf2[:30], title="Word Frequency-Landing Page-Extended URL-Non Leads", 
        path = "../img/WordFreqLPExtURLNNL.jpg")

    string = '''
    Get Dominant Features for Data Frame, no index.
    First set Vectorizer
    Then Set BOW
    Then Count_words()
    then set DF
    '''
    print(string)
    vectorizer_ex = CountVectorizer(analyzer="word", max_features= max_features_extended,
        stop_words=stopwords2, ngram_range=(1,3))
    X_extended = vectorizer_ex.fit_transform(url1Data["extended_url"]).toarray()
    wcex, wfex = count_words(vectorizer_ex, X_extended)
    df_ex = pd.DataFrame(data=X_extended, columns=vectorizer_ex.get_feature_names())
    df = pd.concat([df, df_ex.loc[:,wfex[:10].index]], axis=1)
    print(df)
    print(df.info())