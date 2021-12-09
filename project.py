import numpy as np
import pandas as pd
import sys
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import datetime


def df_year(df):
    data2019 = df.filter(regex='19-').rename(columns=lambda x: x.lstrip('19-'))
    data2020 = df.filter(regex='20-').rename(columns=lambda x: x.lstrip('20-'))
    data2021 = df.filter(regex='21-').rename(columns=lambda x: x.lstrip('21-'))
    return data2019, data2020, data2021


def main():
    # Todo
    # Separate dataframe, make a separate one for month and one for other statistics

    intervention = pd.read_csv(sys.argv[1])
    intervention = intervention.loc[intervention['Intervention category'] == 'Travel']

    # - Graph of number of people travelling by month aka the whole year(jan-dec on x axis)
    monthlyTravel = pd.read_csv(sys.argv[2])

    resident_col = ['Trips by Canadian residents', 'Trips by United States residents',
                    'Trips by all other countries residents', 'Total']

    travel_resident = monthlyTravel.loc[monthlyTravel['Method of Travel'].isin(resident_col)].rename(
        columns={'Method of Travel': 'Traveller Residency'}).set_index('Traveller Residency')

    data2019, data2020, data2021 = df_year(travel_resident)

    total2019 = data2019.to_numpy()[3]
    total2020 = data2020.to_numpy()[3]
    total2021 = data2021.to_numpy()[3]

    print("<2019>", total2019, "\n<2020>", total2020, "\n<2021>", total2021)

    print(travel_resident)

    # - Graph of how people are travelling by month (jan-dec on x axis)

    # - Graph to compare the traveller numbers 2019-2020 vs 2020-2021
    monthlyChange = pd.read_csv(sys.argv[3])

    # - Use tests to check the validity of the data (p-value):
    #  expecting that it could show very little correlation because of how crazy the values differ
    # 1. Use print(stats.normaltest(xa).pvalue) against all 4 traveller number values

    #Need to transpose to make these lines work
    travel_resident=travel_resident.T

    print("Transpose",travel_resident,"\n")

    print(stats.normaltest(travel_resident['Trips by Canadian residents']).pvalue)
    print(stats.normaltest(travel_resident['Trips by United States residents']).pvalue)
    print(stats.normaltest(travel_resident['Trips by all other countries residents']).pvalue)
    print(stats.normaltest(travel_resident['Total']).pvalue)

    # 2. Use print(stats.normaltest(xa).pvalue) against the 4 percentage change

    # - Use one of machine learning methods to compute future monthly values
    # 1. Use polynomial Regression with Degree 3 to calculate future values




    # Example
    # poly = PolynomialFeatures(degree=3, include_bias=True)
    # X_poly = poly.fit_transform(X)
    # model = LinearRegression(fit_intercept=False)
    # model.fit(X_poly, y)

    ##Since needed to change how we get the list of dates. However they do not work with strftime as of now

    dates = travel_resident.index.values.tolist()
    dates = int(dates.strftime("%Y%m"))
    poly = PolynomialFeatures(degree=3, include_bias=True)
    X_poly = poly.fit_transform(dates)
    y = travel_resident['Trips by Canadian residents']
    model = LinearRegression(fit_intercept=False)
    model.fit(X_poly, y)

    y = travel_resident['Trips by United States residents']
    model = LinearRegression(fit_intercept=False)
    model.fit(X_poly, y)

    y = travel_resident['Trips by all other countries residents']
    model = LinearRegression(fit_intercept=False)
    model.fit(X_poly, y)

    y = travel_resident['Total']
    model = LinearRegression(fit_intercept=False)
    model.fit(X_poly, y)

    # - Graph of number of people travelling by month with MC values(jan-dec on x axis)

    # - Check the validity of the data again (p-value)




if __name__ == '__main__':
    main()
