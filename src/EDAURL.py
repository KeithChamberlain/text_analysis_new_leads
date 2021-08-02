import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from MineURL import ManyURL
from makedv import get_search
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.decomposition import NMF


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

def plot_bar_two_cat(category1, category2, ranger=30, title = None, path = "../img/WordFreq_.jpg", columns=[]):
    categories = pd.concat([category1.sort_index(), category2.sort_index()], axis=1)
    categories.columns = columns
    width=0.35
    categories.sort_values(by = columns[0], ascending=False, inplace=True)
    
    fig, ax = plt.subplots(figsize=(16,5))
    ax.bar(x=np.array(range(ranger))-width/2, height=categories.iloc[:ranger,0], width=0.35, label="New Lead")
    ax.bar(x=np.array(range(ranger))+width/2, height=categories.iloc[:ranger,1], width=0.35, label= "Not Lead")
    plt.xticks(range(ranger), categories[:ranger].index, rotation=90)
    ax.set_title(title, fontsize=30)
    ax.set_ylabel("Rel. Freq.", fontsize=20)
    ax.set_xlabel("Words", fontsize=20)
    plt.legend()
    fig.savefig(path)
    plt.show()
    return categories, fig, ax



def plot_words(keys, values, title="Word Frequency", path="./img/WordFreq_.jpg", ylim = None):
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
    ax.set_title(title, fontsize=30)
    ax.set_xlabel('Words', fontsize=20)
    ax.set_ylabel('Rel. Freq.', fontsize=20)
    ax.set_ylim(ylim)
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
    n_features_to_df = 10
    n_features_to_graph = 30
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
        '64','gclid','bwc','campaign','ad','fba','utm','fbclid','wcb','tsr','bwe','near','provided'})
    stopwords2 = set(stopwords.words('english')).union(stopwords_)

    string = '''
    Read Data, Clean/Verify Input
    '''
    print(string)
    df = pd.read_csv("../data/cleaned1.csv", compression="gzip")
    df["search_query"] = df["Search Query"]
    df["tracking_source"] = df["Tracking Source"]
    df.drop(columns=["Likelihood","Is2020","Duration", "Day", "Hour of Day","Search Query", 
        "Tracking Source", "dur_min"],
        inplace=True)
    df['dt'] = df['dt'].apply(pd.to_datetime)
    df['year'] = pd.DatetimeIndex(df['dt']).year
    df['month'] = pd.DatetimeIndex(df['dt']).month
    #df['month_name'] = pd.DatetimeIndex(df['dt']).month_name
    df['day'] = pd.DatetimeIndex(df['dt']).day_of_week
    #df['day_of_week'] = pd.DatetimeIndex(df['dt']).day_name
    df['hour'] = pd.DatetimeIndex(df['dt']).hour

    # Fill NAs
    print(df.info())    
    df["Page"] = df.loc[:,"Page"].fillna(" ").copy()
    print("Search Query Not NA: ", df["search_query"].notna().count())
    df["search_query"] = df.loc[:,"search_query"].fillna(" ").copy()

    print(df["tracking_source"].value_counts())
    
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
    vectorizer_extendedNL = CountVectorizer(analyzer="word", 
        max_features = max_features_extended,
        stop_words=stopwords2, ngram_range=(1,3))
    X_extendedNL = vectorizer_extendedNL.fit_transform(url1Data.loc[index, 
        "extended_url"]).toarray()
    wc1, wf1 = count_words(vectorizer_extendedNL, X_extendedNL)

    vectorizer_extendedNNL = CountVectorizer(analyzer="word", 
        max_features = max_features_extended, 
        stop_words=stopwords2, ngram_range=(1,3))
    X_extendedNNL = vectorizer_extendedNNL.fit_transform(url1Data.loc[index2, 
        "extended_url"]).toarray()
    wc2, wf2 = count_words(vectorizer_extendedNNL, X_extendedNNL)
    plot_words(wf1.index[:n_features_to_graph], wf1[:n_features_to_graph], 
        title="Word Frequency-Landing Page-Extended URL-New Leads", 
        path = "../img/WordFreqLPExtURLNL.jpg",
        ylim=(0,6))
    plot_words(wf2.index[:n_features_to_graph], wf2[:n_features_to_graph], 
        title="Word Frequency-Landing Page-Extended URL-Non Leads", 
        path = "../img/WordFreqLPExtURLNNL.jpg",
        ylim=(0,6))

    plot_bar_two_cat(category1=wf1, category2=wf2, ranger=30, 
        title="Word Frequency-Landing Page-Extended URL", 
        path="../img/WordFreqExtURL_.jpg", columns = ["NewLead","NonLead"])
    

    string = '''
    Get Dominant Features for Data Frame.
    First set Vectorizer
    Then Set BOW
    Then Count_words()
    then set DF
    '''
    print(string)
    vectorizer_ex = CountVectorizer(analyzer="word", 
        max_features= max_features_extended,
        stop_words=stopwords2, ngram_range=(1,3))
    X_extended = vectorizer_ex.fit_transform(url1Data["extended_url"]).toarray()
    wcex, wfex = count_words(vectorizer_ex, X_extended)

    wcindex= np.sum(wcex > 1000)
    df_ex = pd.DataFrame(data=X_extended, columns=vectorizer_ex.get_feature_names())
    df = pd.concat([df, df_ex.loc[:,wcex[:wcindex].index]], axis=1)
    print(df)
    print(df.info())


    string = '''
    Get BOW & Word Counts (wc) for 
    Base URL
    '''
    print(string)
    vectorizer_baseNL = CountVectorizer(analyzer="word", 
        max_features = max_features_base, 
        stop_words=stopwords2, ngram_range=(1,1))
    X_baseNL = vectorizer_baseNL.fit_transform(url1Data.loc[index, 
        "base_url"]).toarray()
    wc1, wf1 = count_words(vectorizer_baseNL, X_baseNL)

    vectorizer_baseNNL = CountVectorizer(analyzer="word", 
        max_features = max_features_base, 
        stop_words=stopwords2, ngram_range=(1,1))
    X_baseNNL = vectorizer_baseNNL.fit_transform(url1Data.loc[index2, 
        "base_url"]).toarray()
    wc2, wf2 = count_words(vectorizer_baseNNL, X_baseNNL)
    
    plot_words(wf1.index[:n_features_to_graph], wf1[:n_features_to_graph], 
        title="Word Frequency-Landing Page-Base URL-New Leads", 
        path = "../img/WordFreqLPBaeURLNL.jpg",
        ylim=(0,18))
    plot_words(wf2.index[:n_features_to_graph], wf2[:n_features_to_graph], 
        title="Word Frequency-Landing Page-Base URL-Non Leads", 
        path = "../img/WordFreqLPBaseURLNNL.jpg",
        ylim=(0,18))
    
    plot_bar_two_cat(category1=wf1, category2=wf2, ranger=30, 
        title="Word Frequency-Landing Page-Base URL", 
        path="../img/WordFreqBaseURL_.jpg", columns = ["NewLead","NonLead"])


    string = '''
    Get BOW & Word Counts (wc) for 
    Search String
    '''
    print(string)
    vectorizer_searchNL = CountVectorizer(analyzer="word", 
        max_features = max_features_search, 
        stop_words=stopwords2, ngram_range=(1,3))
    X_searchNL = vectorizer_searchNL.fit_transform(df.loc[index, 
        "search_query"]).toarray()
    wc1, wf1 = count_words(vectorizer_searchNL, X_searchNL)

    vectorizer_searchNNL = CountVectorizer(analyzer="word", 
        max_features = max_features_search, 
        stop_words=stopwords2, ngram_range=(1,3))
    X_searchNNL = vectorizer_searchNNL.fit_transform(df.loc[index2, 
        "search_query"]).toarray()
    wc2, wf2 = count_words(vectorizer_searchNNL, X_searchNNL)
    plot_words(wf1.index[1:n_features_to_graph], wf1[1:n_features_to_graph], 
        title="Word Frequency-Search-New Leads", 
        path = "../img/WordFreqSearchNL.jpg",
        ylim=(0,6))
    plot_words(wf2.index[1:n_features_to_graph], wf2[1:n_features_to_graph], 
        title="Word Frequency-Search-Non Leads", 
        path = "../img/WordFreqSearchNNL.jpg",
        ylim=(0,6))
    
    
    plot_bar_two_cat(category1=wf1, category2=wf2, ranger=30, 
        title="Word Frequency-Search String", 
        path="../img/WordFreqSearch_.jpg", columns = ["NewLead","NonLead"])
    
    

    string = '''
    Get Dominant Features for base URL Data Frame.
    First set Vectorizer
    Then Set BOW
    Then Count_words()
    then set DF
    '''
    print(string)
    vectorizer_base = CountVectorizer(analyzer="word", 
        max_features= max_features_base,
        stop_words=stopwords2, ngram_range=(1,1))
    X_base = vectorizer_base.fit_transform(url1Data["base_url"]).toarray()
    wcbase, wfbase = count_words(vectorizer_base, X_base)
   
    wcindex= np.sum(wcbase > 150)

    df_base = pd.DataFrame(data=X_base, 
        columns=vectorizer_base.get_feature_names())
    df = pd.concat([df, df_base.loc[:,wcbase[:wcindex].index]], axis=1)
    print(df)
    print(df.info())

    string = '''
    Get Dominant Freatures for search_query DataFrame
    First set Vectoriser
    Then Set BOW
    The Count_words()
    then set DF
    '''
    print(string)

    vectorizer_search = CountVectorizer(analyzer="word", 
        max_features= max_features_search,
        stop_words=stopwords2, ngram_range=(1,3))
    X_search  = vectorizer_search.fit_transform(df["search_query"]).toarray()
    wcsearch, wfsearch = count_words(vectorizer_search, X_search)
    wcindex = np.sum(wcsearch > 150)
    #print(wcsearch[:50])

    # Add an underscore to keep columns seperate
    columns = wcsearch.index
    columns = [_ + "_" for _ in columns]
    wcsearch.index = columns
    features = vectorizer_search.get_feature_names()
    features = [_+"_" for _ in features]
    #print(features)
    df_search = pd.DataFrame(data=X_search, 
        columns=features)
    df = pd.concat([df, df_search.loc[:,wcsearch[:wcindex].index]], axis=1)

    df.drop(columns=["Referral", "Page", "Last URL", "search_query"], inplace=True)
    
    print(df.info())

    df.to_csv("../data/cleaned2.csv", compression="gzip", index=False)


