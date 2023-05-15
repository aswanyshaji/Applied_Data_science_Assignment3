# -*- coding: utf-8 -*-
"""
Created on Sun May 10 01:15:44 2023

@author: ASWANY SHAJI
"""

from sklearn import cluster
import errors as err
import numpy as np
from scipy.optimize import curve_fit
import scipy.optimize as opt
import pandas as pd
import pandas as pd
import numpy as np
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
import cluster_tools as ct


def exp_growth(t, scale, growth):
    """
    Computes exponential function with scale and growth as free parameters
    """
    f = scale * np.exp(growth * (t - 1900))
    return f


def read_clean_transpose(filename):
    """ 
    Read excel file into a data frame, clean the data, and transpose 
    """
    data_frame = pd.read_csv(filename)
    # Make 'Country Name' the index
    data_frame.set_index('Country Name', inplace=True)
    # Delete unwanted columns
    data_frame.drop(labels = [ 'Country Code', 'Indicator Name', \
                                   'Indicator Code'], axis = 1, inplace = True)
    # Delete empty rows
    data_frame.dropna(axis = 0, how = 'all', thresh = None, subset = None, \
                                                                inplace = True)
    # Delete empty columns
    data_frame.dropna(axis = 1, how = 'all', thresh = None, subset = None, \
                                                                inplace = True)
    # Transpose the cleaned data
    clean_df_transpose = data_frame.transpose()
    # Set 'Year' as the index of the transposed data frame
    clean_df_transpose = clean_df_transpose.rename_axis('Year')
    return data_frame, clean_df_transpose


def agriculture_forest_clustering_analysis():
    """ 
    This function cluster the Countries based on their agriculture land area
    and forest area
    """
    agriculture.reset_index(inplace = True)
    forest.reset_index(inplace = True)
    agriculture_2020 = agriculture[["Country Name", "2020"]].copy()
    forest_2020 = forest[["Country Name", "2020"]].copy()
    agriculture_2020 = agriculture_2020[agriculture_2020["2020"].notna()]
    forest_2020 = forest_2020[forest_2020["2020"].notna()]
    agriculture_forest = pd.merge(agriculture_2020, forest_2020, on = \
                                                  "Country Name", how = "outer")
    agriculture_forest = agriculture_forest.dropna()
    agriculture_forest = agriculture_forest.rename(columns = \
                                    {"2020_x":"agriculture", "2020_y":"forest"})
    agriculture_forest.to_excel("agr_for2020.xlsx")                                              
    agr_forest_cluster = agriculture_forest[["agriculture", "forest"]].copy()
    """
    perform normalization before clustering   
    """                                           
    agr_forest_cluster, df_min, df_max = ct.scaler(agr_forest_cluster)
    # loop over number of clusters to find out best silhouette score
    for ncluster in range(2, 10):
        # set up the clusterer with the number of expected clusters
        kmeans = cluster.KMeans(n_clusters = ncluster)
        # Fit the data, results are stored in the kmeans object
        kmeans.fit(agr_forest_cluster) # fit done on x,y pairs
        labels = kmeans.labels_
        # extract the estimated cluster centres
        cen = kmeans.cluster_centers_
        # calculate the silhoutte score
        print(ncluster, skmet.silhouette_score(agr_forest_cluster, labels))
    ncluster = 4
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters = ncluster)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(agr_forest_cluster)
    labels = kmeans.labels_
    #create a new column and add cluster number corresponding to each Country
    agriculture_forest["classification"] = labels
    agriculture_forest_sort = agriculture_forest.sort_values("classification")
    #store the data frame to a csv file for detailed analysis
    agriculture_forest_sort.to_csv("cluster_output.csv")
    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_
    """
    Rescale and show cluster centers(Back Scaling)
    """
    scen = ct.backscale(cen, df_min, df_max)
    xc = scen[:, 0]
    yc = scen[:, 1]
    plt.figure(figsize = (8.0, 8.0))
    cm = plt.cm.get_cmap('tab10')
    #plot the real values and real cluster centers
    plt.scatter(agriculture_forest["agriculture"], agriculture_forest["forest"]\
                        , 20, labels, marker = "o", cmap = cm, label ='cluster')
    plt.scatter(xc, yc, c = "k", marker = "d", s = 100)
    plt.title("CLUSTERING ANALYSIS OF AGRICULTURE VS FOREST", fontsize = 18, \
                                                          fontweight = 'bold')
    plt.xlabel("Agriculture land", fontsize = 15, fontweight = 'bold')
    plt.ylabel("Forest Land", fontsize = 15, fontweight = 'bold')
    plt.tight_layout()
    plt.savefig("agr_for.png", dpi = 300)
    plt.show()
    return
    
    
def arable_land_cereal_yield_clustering_analysis():
    """ 
    This function cluster the Countries based on their arable land and 
    cereal yield
    """
    arable_land.reset_index(inplace = True)
    cereal_yield.reset_index(inplace = True)
    arable_land_2020 = arable_land[["Country Name", "2020"]].copy()
    cereal_yield_2020 = cereal_yield[["Country Name", "2020"]].copy()
    arable_land_2020 = arable_land_2020[arable_land_2020["2020"].notna()]
    cereal_yield_2020 = cereal_yield_2020[cereal_yield_2020["2020"].notna()]
    arable_vs_cereal = pd.merge(arable_land_2020,  cereal_yield_2020, on = \
                                                "Country Name", how = "outer")
    arable_vs_cereal = arable_vs_cereal.dropna()
    arable_vs_cereal = arable_vs_cereal.rename(columns = \
                             {"2020_x":"Arable_land", "2020_y":"Cereal_yield"})
    arable_vs_cereal.to_excel("cereal_agr2020.xlsx")                                              
    arable_vs_cereal_cluster = arable_vs_cereal[["Arable_land", "Cereal_yield"]]\
                                                                        .copy()                                              
    # normalise
    arable_vs_cereal_cluster, df_min, df_max = ct.scaler(arable_vs_cereal_cluster)
    for ncluster in range(2, 10):
        # set up the clusterer with the number of expected clusters
        kmeans = cluster.KMeans(n_clusters = ncluster)
        # Fit the data, results are stored in the kmeans object
        kmeans.fit(arable_vs_cereal_cluster) # fit done on x,y pairs
        labels = kmeans.labels_
        # extract the estimated cluster centres
        cen = kmeans.cluster_centers_
        # calculate the silhoutte score
        print(ncluster, skmet.silhouette_score(arable_vs_cereal_cluster, labels))   
    ncluster = 4
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters = ncluster)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(arable_vs_cereal_cluster) # fit done on x,y pairs
    labels = kmeans.labels_
    arable_vs_cereal["classification"] = labels
    arable_vs_cereal_sort = arable_vs_cereal.sort_values("classification")
    arable_vs_cereal_sort.to_csv("cluster_output1.csv")
    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_
    #Rescale and show cluster centers
    scen = ct.backscale(cen, df_min, df_max)
    xc = scen[:, 0]
    yc = scen[:, 1]
    #xcen = cen[:, 0]
    #ycen = cen[:, 1]
    # cluster by cluster
    plt.figure(figsize=(8.0, 8.0))
    cm = plt.cm.get_cmap('tab10')
    plt.scatter(arable_vs_cereal["Arable_land"], arable_vs_cereal["Cereal_yield"],\
                            10, labels, marker = "o", cmap = cm, label ='cluster')
    plt.scatter(xc, yc, c = "k", marker = "d", s = 80)
    plt.title("ARABLE LAND VS CEREAL YIELD", \
                                             fontsize = 18, fontweight = 'bold')
    plt.xlabel("Arable land", fontsize = 15, fontweight = 'bold')
    plt.ylabel("Cereal yield(kg/Hectare)", fontsize = 15, fontweight = 'bold')
    plt.tight_layout()
    plt.savefig("cer_arab.png", dpi = 300)
    plt.show()
    return
    
def india_cereal_yield_fitting_prediction():
    """ 
    This function creates a model for fitting India's Cereal Yield data
    and use this model for predicting future values
    """
    cereal_yield_t.reset_index(inplace = True)
    df_cereal = cereal_yield_t[["Year", "India"]].copy()
    df_cereal["India"] = pd.to_numeric(df_cereal["India"])
    df_cereal["Year"] = pd.to_numeric(df_cereal["Year"])
    print(df_cereal)
    popt, pcorr = opt.curve_fit(exp_growth, df_cereal["Year"], \
                                df_cereal["India"], p0 = [4e8, 0.03])
    print(*popt)
    """ 
    taking the error value
    """
    sigma = np.sqrt(np.diag(pcorr))
    """
    low and up values for error ranges
    """
    low,up = err.err_ranges(df_cereal["Year"],exp_growth,popt,sigma)
    """
    data fitting
    """
    df_cereal["cerel_yield_exp"] = exp_growth(df_cereal["Year"], *popt)
    plt.figure()
    plt.plot(df_cereal["Year"], df_cereal["India"], label = "data")
    plt.plot(df_cereal["Year"], df_cereal["cerel_yield_exp"], label = "fit")
    plt.ylabel("Cereal Yield (kg/Hectare)")
    """
    plot the error ranges in the graph
    """
    
    plt.fill_between(df_cereal["Year"], low, up, alpha = 0.3, label = "Error Ranges")
    plt.title("INDIA - CEREAL YIELD")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ind_cer.png", dpi = 300)
    plt.show()
    
    """
    prediction of 2035
    """ 
    print("2030:", exp_growth(2030, *popt))
    print("2040:", exp_growth(2040, *popt))
    print("2050:", exp_growth(2050, *popt))
    plt.figure()
    plt.title("PREDICTION OF CEREAL YIELD - INDIA :2035")
    pred_year = np.arange(1960, 2035)
    pred_ind = exp_growth(pred_year, *popt)
    plt.plot(df_cereal["Year"], df_cereal["India"], label = "Data")
    plt.plot(pred_year,pred_ind,label = "Prediction")
    plt.ylabel("Cereal Yield (kg/Hectare)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ind_cer_pre.png", dpi = 300)
    plt.show()
    return
    
    
def us_cereal_yield_fitting_prediction(): 
    """ 
    This function creates a model for fitting US's Cereal Yield data
    and use this model for predicting future values
    """
    cereal_yield_t.reset_index(inplace = True)
    df_cereal = cereal_yield_t[["Year", "United States"]].copy()
    df_cereal["United States"] = pd.to_numeric(df_cereal["United States"])
    df_cereal["Year"] = pd.to_numeric(df_cereal["Year"])
    popt, pcorr = opt.curve_fit(exp_growth, df_cereal["Year"],\
                                df_cereal["United States"], p0 = [4e8, 0.03])
    print(*popt)

    sigma = np.sqrt(np.diag(pcorr))
    """
    low and up values for error ranges
    """
    low,up = err.err_ranges(df_cereal["Year"], exp_growth, popt, sigma)
    """
    data fitting
    """
    plt.figure()
    df_cereal["cerel_yield_exp"] = exp_growth(df_cereal["Year"], *popt)
    plt.plot(df_cereal["Year"], df_cereal["United States"], label = "Data")
    plt.plot(df_cereal["Year"], df_cereal["cerel_yield_exp"], label = "Fit")
    plt.ylabel("Cereal Yield (kg/Hectare)")
    
    """
    plot the error ranges in the graph
    """

    plt.fill_between(df_cereal["Year"], low, up, alpha=0.3, label = "Error Ranges")
    plt.title("UNITED STATES - CEREAL YIELD")
    plt.legend(loc = 'upper left')
    plt.tight_layout()
    plt.savefig("us_cer.png", dpi = 300)
    plt.show()
    
    """
    prediction of 2035
    """ 
    plt.figure()
    plt.title("PREDICTION OF CEREAL YIELD USA: 2035")
    print("2030:", exp_growth(2030, *popt))
    print("2040:", exp_growth(2040, *popt))
    print("2050:", exp_growth(2050, *popt))
    pred_year = np.arange(1960, 2035)
    pred_ind = exp_growth(pred_year, *popt)
    plt.plot(df_cereal["Year"], df_cereal["United States"], label = "data")
    plt.plot(pred_year, pred_ind, label = "prediction")
    plt.ylabel("Cereal Yield (kg/Hectare)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("us_cer_pred.png", dpi = 300)
    plt.show()
    return
    

if __name__ == "__main__":
    # Call the function to clean and transpose the data
    cereal_yield, cereal_yield_t = read_clean_transpose('cereal_yield.csv')
    agriculture, agriculture_t = read_clean_transpose('agriculture.csv')
    forest, forest_t = read_clean_transpose('forest.csv')
    arable_land, arable_land_t = read_clean_transpose('arable_land.csv')
    #Call function to perform clustering
    agriculture_forest_clustering_analysis()
    #Call function to perform clustering
    arable_land_cereal_yield_clustering_analysis()
    #call function to perform fitting and prediction
    india_cereal_yield_fitting_prediction()
    #call function to perform fitting and prediction
    us_cereal_yield_fitting_prediction()
    
    
