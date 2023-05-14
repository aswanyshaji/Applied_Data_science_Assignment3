# -*- coding: utf-8 -*-
"""
Created on Sun May 14 01:15:44 2023

@author: hp
"""

import seaborn as sns
from sklearn import cluster
import errors as err
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import pandas as pd
import numpy as np
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
import cluster_tools as ct

def exp_growth(t, scale, growth):
    """ Computes exponential function with scale and growth as free parameters """
    f = scale * np.exp(growth * t)
    return f

def read_clean_transpose(filename):
    """ Read excel file into a data frame, clean the data, and transpose """
    data_frame = pd.read_csv(filename)
    # Make 'Country Name' the index
    #data_frame.set_index('Country Name', inplace=True)
    # Delete unwanted columns
    data_frame.drop(labels=[ 'Indicator Name', 'Indicator Code'], axis=1, inplace=True)
    # Delete empty rows
    data_frame.dropna(axis=0, how='all', thresh=None, subset=None, inplace=True)
    # Delete empty columns
    data_frame.dropna(axis=1, how='all', thresh=None, subset=None, inplace=True)
    # Transpose the cleaned data
    clean_df_transpose = data_frame.transpose()
    # Set 'Year' as the index of the transposed data frame
    clean_df_transpose = clean_df_transpose.rename_axis('Year')
    return data_frame, clean_df_transpose

def agriculture_forest_clustering_analysis():
    #df_agrar = pd.read_csv("agriculture.csv")
    #df_forest = pd.read_csv("agriculture_gdp.csv")
    #print(df_agrar.describe())
    #print(df_forest.describe())
    # drop rows with nan's in 2020
    #agriculture = agriculture[agriculture["2020"].notna()]
    #forest = forest[forest["2020"].notna()]
    #print(df_agrar.describe(
    # alternative way of targetting one or more columns
    #df_forest = df_forest.dropna(subset=["2020"])
    #print(df_forest.describe)
    agriculture_2020 = agriculture[["Country Name", "Country Code", "2020"]].copy()
    forest_2020 = forest[["Country Name", "Country Code", "2020"]].copy()
    agriculture_2020 = agriculture_2020[agriculture_2020["2020"].notna()]
    forest_2020 = forest_2020[forest_2020["2020"].notna()]
    #print(df_agr2020.describe())
    #print(df_for2020.describe())
    agriculture_forest = pd.merge(agriculture_2020, forest_2020, on="Country Name", how="outer")
    #print(df_2020.describe())
    #agriculture_forest.to_excel("agr_for2020.xlsx")
    #print(df_2020.describe())
    agriculture_forest = agriculture_forest.dropna()
    agriculture_forest = agriculture_forest.rename(columns={"2020_x":"agriculture", "2020_y":"forest"})
    agriculture_forest.to_excel("agr_for2020.xlsx")                                              
    agr_forest_cluster = agriculture_forest[["agriculture", "forest"]].copy()                                              
    # entries with one datum or less are useless.
    #print()
    #print(df_2020.describe())
    # rename columns
    #df_2020 = df_2020.rename(columns={"2020_x":"agriculture", "2020_y":"forest"})
    #df_cluster = df_2020[["agriculture", "forest"]].copy()
    # normalise
    agr_forest_cluster, df_min, df_max = ct.scaler(agr_forest_cluster)
    print("n score")
    # loop over number of clusters
    for ncluster in range(2, 10):
        # set up the clusterer with the number of expected clusters
        kmeans = cluster.KMeans(n_clusters=ncluster)
        # Fit the data, results are stored in the kmeans object
        kmeans.fit(agr_forest_cluster) # fit done on x,y pairs
        labels = kmeans.labels_
        # extract the estimated cluster centres
        cen = kmeans.cluster_centers_
        # calculate the silhoutte score
        print(ncluster, skmet.silhouette_score(agr_forest_cluster, labels))
    
    ncluster = 4
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(agr_forest_cluster) # fit done on x,y pairs
    labels = kmeans.labels_
    agriculture_forest["classification"] = labels
    #print(df_2020)
    agriculture_forest_sort = agriculture_forest.sort_values("classification")
    agriculture_forest_sort.to_csv("cluster_output.csv")
    #print(labels)

    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_
    #Rescale and show cluster centers
    scen = ct.backscale(cen, df_min, df_max)
    xc = scen[:,0]
    yc = scen[:,1]
    #xcen = cen[:, 0]
    #ycen = cen[:, 1]
    # cluster by cluster
    plt.figure(figsize=(8.0, 8.0))
    cm = plt.cm.get_cmap('tab10')
    plt.scatter(agriculture_forest["agriculture"], agriculture_forest["forest"], 10, labels, marker="o", cmap=cm, label ='cluster')
    plt.scatter(xc, yc, c="k", marker="d", s=80)
    #plt.scatter(xcen, ycen, 45, "k", marker="d")
    plt.xlabel("agriculture")
    plt.ylabel("forest")
    plt.show()

def arable_land_cereal_yield_clustering_analysis():
    #df_agrar = pd.read_csv("agriculture.csv")
    #df_forest = pd.read_csv("agriculture_gdp.csv")
    #print(df_agrar.describe())
    #print(df_forest.describe())
    # drop rows with nan's in 2020
    #agriculture = agriculture[agriculture["2020"].notna()]
    #forest = forest[forest["2020"].notna()]
    #print(df_agrar.describe(
    # alternative way of targetting one or more columns
    #df_forest = df_forest.dropna(subset=["2020"])
    #print(df_forest.describe)
    agriculture_2020 = agriculture[["Country Name", "Country Code", "2020"]].copy()
    forest_2020 = forest[["Country Name", "Country Code", "2020"]].copy()
    agriculture_2020 = agriculture_2020[agriculture_2020["2020"].notna()]
    forest_2020 = forest_2020[forest_2020["2020"].notna()]
    #print(df_agr2020.describe())
    #print(df_for2020.describe())
    agriculture_forest = pd.merge(agriculture_2020, forest_2020, on="Country Name", how="outer")
    #print(df_2020.describe())
    #agriculture_forest.to_excel("agr_for2020.xlsx")
    #print(df_2020.describe())
    agriculture_forest = agriculture_forest.dropna()
    agriculture_forest = agriculture_forest.rename(columns={"2020_x":"agriculture", "2020_y":"forest"})
    agriculture_forest.to_excel("agr_for2020.xlsx")                                              
    agr_forest_cluster = agriculture_forest[["agriculture", "forest"]].copy()                                              
    # entries with one datum or less are useless.
    #print()
    #print(df_2020.describe())
    # rename columns
    #df_2020 = df_2020.rename(columns={"2020_x":"agriculture", "2020_y":"forest"})
    #df_cluster = df_2020[["agriculture", "forest"]].copy()
    # normalise
    agr_forest_cluster, df_min, df_max = ct.scaler(agr_forest_cluster)
    print("n score")
    # loop over number of clusters
    for ncluster in range(2, 10):
        # set up the clusterer with the number of expected clusters
        kmeans = cluster.KMeans(n_clusters=ncluster)
        # Fit the data, results are stored in the kmeans object
        kmeans.fit(agr_forest_cluster) # fit done on x,y pairs
        labels = kmeans.labels_
        # extract the estimated cluster centres
        cen = kmeans.cluster_centers_
        # calculate the silhoutte score
        print(ncluster, skmet.silhouette_score(agr_forest_cluster, labels))
    
    ncluster = 4
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(agr_forest_cluster) # fit done on x,y pairs
    labels = kmeans.labels_
    agriculture_forest["classification"] = labels
    #print(df_2020)
    agriculture_forest_sort = agriculture_forest.sort_values("classification")
    agriculture_forest_sort.to_csv("cluster_output.csv")
    #print(labels)

    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_
    #Rescale and show cluster centers
    scen = ct.backscale(cen, df_min, df_max)
    xc = scen[:,0]
    yc = scen[:,1]
    #xcen = cen[:, 0]
    #ycen = cen[:, 1]
    # cluster by cluster
    plt.figure(figsize=(8.0, 8.0))
    cm = plt.cm.get_cmap('tab10')
    plt.scatter(agriculture_forest["agriculture"], agriculture_forest["forest"], 10, labels, marker="o", cmap=cm, label ='cluster')
    plt.scatter(xc, yc, c="k", marker="d", s=80)
    #plt.scatter(xcen, ycen, 45, "k", marker="d")
    plt.xlabel("agriculture")
    plt.ylabel("forest")
    plt.show()

def fitting_prediction():
    
    popt, pcorr = opt.curve_fit(exp_growth, df_cereal["Year"], df_cereal["India"], p0=[4e8, 0.03])
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
    plt.plot(df_cereal["Year"], df_cereal["India"], label="data")
    plt.plot(df_cereal["Year"], df_cereal["cerel_yield_exp"], label="fit")
    """
    plot the error ranges in the graph
    """
    plt.fill_between(df_cereal["Year"],low,up,alpha=0.6)
    plt.title("INDIA(Cereal_yield")
    plt.legend()
    plt.show()
    """
    prediction of 2035
    """ 
    plt.figure()
    plt.title("PREDICTION OF Cereal Yield[2035]")
    pred_year = np.arange(1960,2035)
    pred_ind = exp_growth(pred_year,*popt)
    plt.plot(df_cereal["Year"],df_cereal["India"],label="data")
    plt.plot(pred_year,pred_ind,label="prediction")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Call the function to clean and transpose the data
    cereal_yield, cereal_yield_t = read_clean_transpose('cereal_yield.csv')
    agriculture, agriculture_t = read_clean_transpose('agriculture.csv')
    forest, forest_t = read_clean_transpose('forest.csv')
    arable_land, arable_land_t = read_clean_transpose('arable_land.csv')
    agriculture_forest_clustering_analysis()
    arable_land_cereal_yield_clustering_analysis()
    
    
