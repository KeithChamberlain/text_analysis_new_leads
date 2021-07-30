import numpy as np
from numpy.random import PCG64, Generator
import pandas as pd



def boot_ci2(bs, fun=np.mean, ci=0.95, axis = 0):
    '''
    Function assumes bootstrapped results are pre-calculated and 
    all that is needed is to calculate the final function and 
    CI. Axis needs to be provided in case the data come in a 
    different format.
    '''
    bs_stat = fun(bs, axis=axis)
    low_bound = (100. - ci) / 2
    high_bound = 100. - low_bound
    lower_ci_s, upper_ci_s = np.percentile(bs, [low_bound, high_bound], axis = axis)
    
    return pd.DataFrame({"means":bs_stat, "lower_ci_s":lower_ci_s, "upper_ci_s":upper_ci_s})

def boot_ci(bs, fun=np.mean, ci=0.95):
    '''
    Function calculates the confidence interval of a bootstrapped
    sample using the percentile method.
    bs
        a bootstrap array.
    fun
        a function 
    ci
        confidence interval

    returns:
            bootstrapped mean, lower and upper confidence intervals (ci)
    '''
    bs_stat = list(map(fun, bs))
    bs_stat2 = fun(bs_stat, axis=None)
    low_bound = (100. - ci) / 2
    high_bound = 100. - low_bound
    lower_ci, upper_ci = np.percentile(bs, [low_bound, high_bound])
    
    return bs_stat2, lower_ci, upper_ci

def bootstrap(arr, seed = None, iterations = 1000, weights = None, replace=True):
    if type(arr) not in [pd.core.series.Series, np.ndarray, pd.core.frame.DataFrame]:
        arr = np.array(arr)
    
    rows = arr.shape if len(arr.shape) < 2 else arr.shape[0]

    rng = Generator(PCG64(seed=seed))
    
    boot_samples = list()
    for _ in np.arange(iterations):
        boot_samples.append(rng.choice(a = arr, size=rows, 
            replace=replace, p = weights))
    
    return np.array(boot_samples)


if __name__ == "__main__":
    string = '''
    READ DATA
    '''
    print(string)
    df=pd.read_csv("../data/train.csv")
    print(df.info())
    df['dur_min'] = df['dur_min'].fillna(0) # Fill One Missing
    dm = df['dur_min']


    string = '''
    GRAB BOOTSTRAP SAMPLES
    '''
    print(string)
    dmb = bootstrap(dm)


    print("Post bootstrap")
    print(" Shape of bootstraps is :", dmb.shape)

    string = '''
    GRAB BOOTSTRAP MEAN & 95% BOOTSTRAP CIs
    '''
    print(string)
    print(boot_ci(dmb))
