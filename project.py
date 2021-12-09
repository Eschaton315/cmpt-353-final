import numpy as np
import pandas as pd
import sys
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import datetime 


def df_year(df):
    data2019 = df.filter(regex='19-').rename(columns=lambda x: x.lstrip('19-'))
    data2020 = df.filter(regex='20-').rename(columns=lambda x: x.lstrip('20-'))
    data2021 = df.filter(regex='21-').rename(columns=lambda x: x.lstrip('21-'))
    return data2019, data2020, data2021


def df_split(df):
    resident_col = ['Trips by Canadian residents', 'Trips by United States residents',
                    'Trips by all other countries residents', 'Total']
    method_col = ['Automobile', 'Plane', 'Bus', 'Train, boat and other methods']

    travel_resident = df.loc[df['Method of Travel'].isin(resident_col)].rename(
        columns={'Method of Travel': 'Traveller Residency'}).set_index('Traveller Residency')

    travel_method = df.loc[df['Method of Travel'].isin(method_col)].set_index('Method of Travel')

    travel_method = travel_method.groupby('Method of Travel').sum()

    return travel_resident, travel_method


def main():
    # Todo
    # Separate dataframe, make a separate one for month and one for other statistics

    intervention = pd.read_csv(sys.argv[1])
    intervention = intervention.loc[intervention['Intervention category'] == 'Travel']

    # - Graph of number of people travelling by month aka the whole year(jan-dec on x axis)
    monthlyTravel = pd.read_csv(sys.argv[2])

    travel_resident, travel_method = df_split(monthlyTravel)

    data2019, data2020, data2021 = df_year(travel_resident)

    print(travel_resident)
    print(travel_method)

    # - Graph of how people are travelling by month (jan-dec on x axis)

    # - Graph to compare the traveller numbers 2019-2020 vs 2020-2021
    monthlyChange = pd.read_csv(sys.argv[3])

 
    # - Use tests to check the validity of the data (p-value):
    #  expecting that it could show very little correlation because of how crazy the values differ
    # 1. Use print(stats.normaltest(xa).pvalue) against all 4 traveller number values
    print(stats.normaltest(travel_resident['Trips by Canadian residents']).pvalue)
    print(stats.normaltest(travel_resident['Trips by United States residents']).pvalue)
    print(stats.normaltest(travel_resident['Trips by all other countries residents']).pvalue)
    print(stats.normaltest(travel_resident['Total']).pvalue)
    
    # 2. Use print(stats.normaltest(xa).pvalue) against the 4 percentage change
    

    # - Use one of machine learning methods to compute future monthly values
    # 1. Use polynomial Regression with Degree 3 to calculate future values
    
    #Example
# poly = PolynomialFeatures(degree=3, include_bias=True)
# X_poly = poly.fit_transform(X)
# model = LinearRegression(fit_intercept=False)
# model.fit(X_poly, y)
    dates = travel_resident['Traveller Residency']
    dates = int(dates.strftime("%Y%m"))
    poly = PolynomialFeatures(degree = 3, include_bias = True)
    X_poly = poly.fit_transform(dates)
    y = travel_resident['Trips by Canadian residents']
    model = LinearRegression(fit_intercept = False)
    model.fit(X_poly, y)
    
       
    y = travel_resident['Trips by United States residents']
    model = LinearRegression(fit_intercept = False)
    model.fit(X_poly, y)
    

    y = travel_resident['Trips by all other countries residents']
    model = LinearRegression(fit_intercept = False)
    model.fit(X_poly, y)
    

    y = travel_resident['Total']
    model = LinearRegression(fit_intercept = False)
    model.fit(X_poly, y)

    # - Graph of number of people travelling by month with MC values(jan-dec on x axis)

    # - Check the validity of the data again (p-value)

    # print(df)


if __name__ == '__main__':
    main()
