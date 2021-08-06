import numpy as np
import pandas as pd
import scipy.stats as stats
from numpy.random import PCG64, Generator, beta, random
from numpy import random
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from boot_ import bootstrap, boot_ci2

def fit_dist(bs, distribution, params = {"floc":0, "fscale":100}, 
    sum_stat=np.mean):
    '''
    General function for fitting distributions in the scipy.stats library.
    bs = the (potentially bootstrapped) data.
    distribution = distribution function.
    params = a dictionary of parameters/commands for the distribution.
    sum_stat = summary statistic. use only if bootstrapped, otherwise,
        the summary stat will mess up the result of the function.
    return: summary of fit distributions (if bs has more than one dimension;
        or obj, the results of the single fitted distribution with no 
        summary statistics applied. 
    '''
    fun = distribution
    rows = 1 if len(bs.shape) < 2 else bs.shape[0]
    obj = list()
    print("The number of rows is: ", rows)
    for _ in range(rows):
        
        if rows == 1:
            dist_to_fit = fun(bs, **params)
            if _ % 500 == 0:
                print(_, " ", dist_to_fit)
            obj.append(dist_to_fit)
        else:
            dist_to_fit = fun(bs[:,_], **params)
            if _ % 500 == 0:
                print(_, " ", dist_to_fit)
            obj.append(dist_to_fit)
    obj = np.array(obj)
    
    result = sum_stat(obj, axis=0) if sum_stat is not None else obj
    return result

def get_draws(distribution=np.random.beta, params = {"a":0.1, "b":0.1, "size":10000}, seed = None):
    '''
    Wrapper for np.random functions.
    distrubiton = np.random function.
    params = parameters for the distribution in a dictionary.
    seed = optional seed for random values (not tested yet)
    returns: draws from a distribution with the given parameters.
    '''
    rng = np.random.Generator(np.random.PCG64(seed=seed))
    draws =  distribution(**params)
    return draws


class FitDistrib():
    '''
    Class FitDistrib takes in data and fits an a-priori distribution
    either by bootstrapped samples or by a single fit. 
    '''
    def __init__(self, column_to_fill, fit_distribution=stats.beta.fit, 
        fill_distribution = random.beta,
        bootstrap=True, size=10000, replace_dict={0.0000:.000001, 100.0000:99.999999}, 
        fit_params={"floc":0, "fscale":100}, draw_params={"a":0.1, "b":.1, "size":10000}, 
        seed = None):

        self.column_to_fill = column_to_fill
        self.fit_distribution = fit_distribution
        self.fill_distribution = fill_distribution
        self.bootstrap = bootstrap
        self.size = size
        self.replace_dict = replace_dict
        self.fit_params = fit_params
        self.draw_params = draw_params
        self.seed = seed
        self.print_miss()
        self.clean_distrib()
        self.fit_distrib()
        self.interpolate()


    def print_miss(self):
        '''
        Status function prints some helper information for users.
        '''
        self.missing = self.column_to_fill.isna().copy()
        self.not_missing = ~self.missing.copy()
        string = f"""
        The Column {self.column_to_fill.name} has {self.missing.sum()} missing values, and 
        {self.not_missing.sum()} filled values, for a total of {self.missing.sum()+
        self.not_missing.sum()}
        rows. 
        
        This compares to the expected shape of {self.column_to_fill.shape[0]} with
        {self.column_to_fill.shape[0]-self.missing.sum()-self.not_missing.sum()} unaccounted for 
        records. 

        Attempting to fill {self.missing.sum()} missing records with distributional assumptions
        using the {self.fill_distribution} distribution.
        """
        print(string)
        return None

    def clean_distrib(self):
        '''
        Meant to take care of any cleaning before fitting, such as
        scaling or tapering extreme values. For the beta distribution, 
        it may look like replacing values of 0 and 100 with near values 
        not equal to 0 or 100.
        '''
        if self.replace_dict:
            for key, value in self.replace_dict.items():
                self.column_to_fill.replace(to_replace=key, 
                    value=value, inplace=True)
        return None

    def fit_distrib(self):
        '''
        Workhorse function. Fits bootstrapped samples (if requested), and fills in
        two arguements from the distributions based on fitted results. 
        '''
        ss = self.column_to_fill.copy()
        ss = ss[ss.notna()].copy()
        if self.bootstrap: # Then grab bootstrapped sample
            bs = bootstrap(ss, iterations=self.size)
            fits_ = fit_dist(bs, self.fit_distribution, self.fit_params, sum_stat=None)
            self.distrib_parameters = boot_ci2(fits_)
            i=0
            for key in self.draw_params.keys():
                self.draw_params[key] = self.distrib_parameters.iloc[i,0]
                i+=1
                if i==2:
                    break
            self.draw_params["size"] = self.missing.sum()
        else:
            self.distrib_parameters = fit_dist(ss, distribution = self.fit_distribution, 
                params = self.fit_params, sum_stat=None)
            i = 0
            for key in self.draw_params.keys():
                self.draw_params[key] = self.distrib_parameters[0][i]
                i+=1
                if i == 2:
                    break
            self.draw_params["size"] = self.missing.sum()

        string = f'''
        The Fitted Distribution Parameters are: 
        {self.draw_params}.'''
        print(string)  
        return None

    def interpolate(self):
        '''
        Fill in the missing values.
        '''
        draws = get_draws(distribution = self.fill_distribution, params=self.draw_params, seed=self.seed) * \
            (self.fit_params["fscale"] if self.fit_params["fscale"] else 1)
        self.column_to_fill.loc[self.missing] = draws
        return None

    def get_data(self):
        return self.column_to_fill.copy()

    def plot_dists(self, filled_data, original_data, 
        xlab="Data (with missing) and Data Filled Distributions", xaxsize=25, col1=None, col2=None,
        path=None):
        '''
        Figure for distributions
        '''
        fig, ax = plt.subplots(figsize=(16,5))
        ax.hist(filled_data, bins=75, alpha=0.5, density=True, color=col1, label = "Filled")
        ax.hist(original_data, bins=75, alpha=0.5, density=True, color=col2, label = "Original")
        ax.set_title("Histograms of Filled and Original Data", fontsize=35)
        ax.set_ylabel("Density", fontsize=25)
        ax.set_xlabel(xlab, fontsize=xaxsize)
        plt.legend()
        if path is not None:
            fig.savefig(path)
        plt.show()
        


if __name__ == "__main__":
    '''
    Interpolate Likelihood Missing Values by sampling
    from beta distribution (default).
    '''
    df = pd.read_csv("../data/cleaned1.csv", compression="gzip", low_memory=False)
    filled = FitDistrib(df["Likelihood"].copy(), bootstrap=False)
    df["Likelihood_fill"] = filled.get_data()
    print(df.info())
    filled.plot_dists(df["Likelihood_fill"], df["Likelihood"])
    
    '''
    Interpolate Hang_Voice Missing Values by sampling from 
    binomial distribution with probabilities set to match class
    ratios.
    '''
    vc = df["Hang_Voice"].value_counts()
    vcf = vc/vc.sum()
    print(vc)
    print(vcf)
    

