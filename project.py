import numpy as np
import pandas as pd
import sys
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import datetime as dt


def df_year(df):
    data2019 = df.filter(regex='19-').rename(columns=lambda x: x.lstrip('19-'))
    data2020 = df.filter(regex='20-').rename(columns=lambda x: x.lstrip('20-'))
    data2021 = df.filter(regex='21-').rename(columns=lambda x: x.lstrip('21-'))
    return data2019, data2020, data2021


def main():
    # Todo
    # Separate dataframe, make a separate one for month and one for other statistics

    intervention = pd.read_csv(sys.argv[1], parse_dates = ['Date'])
    intervention = intervention.loc[intervention['Intervention category'] == 'Travel']

    # - Graph of number of people travelling by month aka the whole year(jan-dec on x axis)
    #monthlyTravel = pd.read_csv(sys.argv[2], parse_dates = ['Date'])
    monthlyTravel = pd.read_csv(sys.argv[2])
    resident_col = ['Trips by Canadian residents', 'Trips by United States residents',
                    'Trips by all other countries residents', 'Total']

    travel_resident = monthlyTravel.loc[monthlyTravel['Method of Travel'].isin(resident_col)].rename(
        columns={'Method of Travel': 'Traveller Residency'}).set_index('Traveller Residency')

    data2019, data2020, data2021 = df_year(travel_resident)

    total2019 = data2019.to_numpy()[3]
    total2020 = data2020.to_numpy()[3]
    total2021 = data2021.to_numpy()[3]

    # - Graph of how people are travelling by month (jan-dec on x-axis)
    month_name = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"]

    plt.figure(figsize=(9, 7))
    plt.plot(month_name, total2019, '-b', label='2019')
    plt.plot(month_name, total2020, '-r', label='2020')
    plt.plot(month_name[:len(total2021)], total2021, '-g', label='2021')
    plt.ticklabel_format(style='plain', axis='y')
    plt.yticks(np.arange(0, 12000000, 500000))
    plt.title("How people are Travelling by Month")
    plt.legend()
    plt.ylabel("# of People Travelling to Canada")
    plt.xlabel("Month")
    plt.ylim(0, 11000000)
    # plt.show()
    plt.savefig('monthly_travel.png')

    # - Graph to compare the traveller numbers 2019-2020 vs 2020-2021
    
    monthlyChange = pd.read_csv(sys.argv[3])

    travel_change = monthlyChange.loc[monthlyChange['Method of Travel'].isin(resident_col)].rename(
        columns={'Method of Travel': 'Traveller Residency'}).set_index('Traveller Residency')

    print(travel_change)

    # - Use tests to check the validity of the data (p-value):
    #  expecting that it could show very little correlation because of how crazy the values differ
    # 1. Use print(stats.normaltest(xa).pvalue) against all 4 traveller number values


    # Need to transpose to make these lines work

    travel_resident = travel_resident.T

    # print("Transpose", travel_resident, "\n")

    print(stats.normaltest(travel_resident['Trips by Canadian residents']).pvalue)
    print(stats.normaltest(travel_resident['Trips by United States residents']).pvalue)
    print(stats.normaltest(travel_resident['Trips by all other countries residents']).pvalue)
    print(stats.normaltest(travel_resident['Total']).pvalue)

    # 2. Use print(stats.normaltest(xa).pvalue) against the 4 percentage change

    # - Use one of machine learning methods to compute future monthly values
    # 1. Use polynomial Regression with Degree 3 to calculate future values

    dates = pd.read_csv(sys.argv[3])
    datesPredict = pd.read_csv(sys.argv[4])
   
    poly = PolynomialFeatures(degree = 3, include_bias=True)
    X_poly = poly.fit_transform(dates)
    y = travel_resident['Trips by Canadian residents']
    modelCan = LinearRegression(fit_intercept=False)
    modelCan.fit(X_poly, y)
    yCan_pred = modelCan.predict(datesPredict)

    y = travel_resident['Trips by United States residents']
    modelUS = LinearRegression(fit_intercept=False)
    modelUS.fit(X_poly, y)
    modelUS.fit(X_poly, y)
    yUS_pred = modelUS.predict(datesPredict)
    
    y = travel_resident['Trips by all other countries residents']
    modelOther = LinearRegression(fit_intercept=False)
    modelOther.fit(X_poly, y)
    modelOther.fit(X_poly, y)
    yOther_pred = modelOther.predict(datesPredict)

    y = travel_resident['Total']
    modelTotal = LinearRegression(fit_intercept=False)
    modelTotal.fit(X_poly, y)
    modelTotal.fit(X_poly, y)
    yTotal_pred = modelTotal.predict(datesPredict)

    # - Graph of number of people travelling by month with MC values(jan-dec on x axis)

    # - Check the validity of the data again (p-value)



if __name__ == '__main__':
    main()
