import numpy as np
import pandas as pd
import sys
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import datetime as dt


def df_year(df, date_str):
    data = df.filter(regex=date_str).rename(columns=lambda x: x.strip(date_str))
    return data


def main():
    # Todo
    # Separate dataframe, make a separate one for month and one for other statistics

    intervention = pd.read_csv(sys.argv[1], parse_dates=['Date'])
    intervention = intervention.loc[intervention['Intervention category'] == 'Travel']
    interventionPresent = intervention.drop(columns = ['Entry ID', 'Jurisdiction', 'Intervention category', 'Intervention type', 'Organization'])
    print(interventionPresent)

    # - Graph of number of people travelling by month aka the whole year(jan-dec on x axis)

    monthlyTravel = pd.read_csv(sys.argv[2])
    resident_col = ['Trips by Canadian residents', 'Trips by United States residents',
                    'Trips by all other countries residents', 'Total']

    travel_resident = monthlyTravel.loc[monthlyTravel['Method of Travel'].isin(resident_col)].rename(
        columns={'Method of Travel': 'Traveller Residency'}).set_index('Traveller Residency')

    datestr = travel_resident.columns.values.tolist()
    datestr_after_covid = datestr[16:]

    data2019 = df_year(travel_resident, '19-')
    data2020 = df_year(travel_resident, '20-')
    data2021 = df_year(travel_resident, '21-')

    total2019 = data2019.to_numpy()[3]
    total2020 = data2020.to_numpy()[3]
    total2021 = data2021.to_numpy()[3]

    # - Graph of how people are travelling by month (jan-dec on x-axis)
    month_name = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig1, ax1 = plt.subplots(figsize=(9, 7))
    ax1.plot(month_name, total2019, '-b', label='2019')
    ax1.plot(month_name, total2020, '-r', label='2020')
    ax1.plot(month_name[:len(total2021)], total2021, '-g', label='2021')
    plt.ticklabel_format(style='plain', axis='y')
    plt.yticks(np.arange(0, 12000000, 500000))
    ax1.title.set_text("How people are Travelling by Month")
    plt.legend()
    ax1.set_ylabel("# of People Travelling to Canada")
    ax1.set_xlabel("Month")
    ax1.set_ylim(0, 11000000)
    # plt.show()
    plt.savefig('monthly_travel.png')

    # - Graph to compare the traveller numbers 2019-2020 vs 2020-2021

    monthlyChange = pd.read_csv(sys.argv[3])

    travel_change = monthlyChange.loc[monthlyChange['Method of Travel'].isin(resident_col)].rename(
        columns={'Method of Travel': 'Traveller Residency'}).set_index('Traveller Residency')

    change2019_2020 = df_year(travel_change, '2019-2020')
    change2020_2021 = df_year(travel_change, '2020-2021')
    total_change2019_2020 = change2019_2020.to_numpy()[3]
    total_change2020_2021 = change2020_2021.to_numpy()[3]

    fig2, ax2 = plt.subplots()
    ax2.plot(month_name, total_change2019_2020, '-b', label='2019-2020')
    ax2.plot(month_name[:len(total_change2020_2021)], total_change2020_2021, '-r', label='2020-2021')
    plt.yticks(np.arange(-100, 250, 25))
    # need title
    ax2.set_ylabel("Change %")
    ax2.set_xlabel("Month")
    plt.legend()
    plt.savefig('monthly_change.png')

    # - Use tests to check the validity of the data (p-value):
    #  expecting that it could show very little correlation because of how crazy the values differ
    # 1. Use print(stats.normaltest(xa).pvalue) against all 4 traveller number values

    # Need to transpose to make these lines work

    travel_resident = travel_resident.T

    # print("Transpose", travel_resident, "\n")

    print("pvalue for stats with pre covid data \n")
    print(stats.normaltest(travel_resident['Trips by Canadian residents']).pvalue)
    print(stats.normaltest(travel_resident['Trips by United States residents']).pvalue)
    print(stats.normaltest(travel_resident['Trips by all other countries residents']).pvalue)
    print(stats.normaltest(travel_resident['Total']).pvalue)

    travel_residentOnlyAfterCovid = travel_resident.iloc[16:]

    print("pvalue for stats without pre covid data \n")
    print(stats.normaltest(travel_residentOnlyAfterCovid['Trips by Canadian residents']).pvalue)
    print(stats.normaltest(travel_residentOnlyAfterCovid['Trips by United States residents']).pvalue)
    print(stats.normaltest(travel_residentOnlyAfterCovid['Trips by all other countries residents']).pvalue)
    print(stats.normaltest(travel_residentOnlyAfterCovid['Total']).pvalue)

    # - Use one of machine learning methods to compute future monthly values
    # 1. Use polynomial Regression with Degree 3 to calculate future values

    # Without precovid data

    dates = pd.read_csv(sys.argv[4])
    datesOnlyAfterCovid = dates.iloc[16:]
    datesPredict = pd.read_csv(sys.argv[5])

    # With pre-covid data
    X_poly = dates
    y = travel_resident['Trips by Canadian residents']
    modelCan = LinearRegression(fit_intercept=False)
    modelCan.fit(X_poly, y)
    yCan_pred = modelCan.predict(datesPredict)
    # print(yCan_pred)

    y = travel_resident['Trips by United States residents']
    modelUS = LinearRegression(fit_intercept=False)
    modelUS.fit(X_poly, y)
    modelUS.fit(X_poly, y)
    yUS_pred = modelUS.predict(datesPredict)
    # print(yUS_pred)

    y = travel_resident['Trips by all other countries residents']
    modelOther = LinearRegression(fit_intercept=False)
    modelOther.fit(X_poly, y)
    modelOther.fit(X_poly, y)
    yOther_pred = modelOther.predict(datesPredict)
    # print(yOther_pred)

    y = travel_resident['Total']
    modelTotal = LinearRegression(fit_intercept=False)
    modelTotal.fit(X_poly, y)
    modelTotal.fit(X_poly, y)
    yTotal_pred = modelTotal.predict(datesPredict)
    # print(yTotal_pred)

    # Without pre-covid data
    X_poly = datesOnlyAfterCovid
    y = travel_residentOnlyAfterCovid['Trips by Canadian residents']
    modelCanWC = LinearRegression(fit_intercept=False)
    modelCanWC.fit(X_poly, y)
    yCanWC_pred = modelCanWC.predict(datesPredict)
    # print(yCanWC_pred)

    y = travel_residentOnlyAfterCovid['Trips by United States residents']
    modelUSWC = LinearRegression(fit_intercept=False)
    modelUSWC.fit(X_poly, y)
    modelUSWC.fit(X_poly, y)
    yUSWC_pred = modelUSWC.predict(datesPredict)
    # print(yUSWC_pred)

    y = travel_residentOnlyAfterCovid['Trips by all other countries residents']
    modelOtherWC = LinearRegression(fit_intercept=False)
    modelOtherWC.fit(X_poly, y)
    modelOtherWC.fit(X_poly, y)
    yOtherWC_pred = modelOtherWC.predict(datesPredict)
    # print(yOtherWC_pred)

    y = travel_residentOnlyAfterCovid['Total']
    modelTotalWC = LinearRegression(fit_intercept=False)
    modelTotalWC.fit(X_poly, y)
    modelTotalWC.fit(X_poly, y)
    yTotalWC_pred = modelTotalWC.predict(datesPredict)
    # print(yTotalWC_pred)

    # - Graph of number of people travelling by month with MC values(jan-dec on x axis)

    # - Check the validity of the data again (p-value)

    datestr_predict = ["21-Oct", "21-Nov", "21-Dec", "22-Jan", "22-Feb", "22-Mar", "22-Apr", "22-May", "22-Jun",
                       "22-Jul", "22-Aug", "22-Sep", "22-Oct", "22-Nov", "22-Dec"]
    predictedData = pd.DataFrame(columns=['dates', 'Trips by Canadian residents', 'Trips by United States residents',
                                          'Trips by all other countries residents', 'Total'])
    predictedData['dates'] = np.append(datestr, datestr_predict)
    predictedData['Trips by Canadian residents'] = np.append(travel_resident['Trips by Canadian residents'], yCan_pred)
    predictedData['Trips by United States residents'] = np.append(travel_resident['Trips by United States residents'],
                                                                  yUS_pred)
    predictedData['Trips by all other countries residents'] = np.append(
        travel_resident['Trips by all other countries residents'], yOther_pred)
    predictedData['Total'] = np.append(travel_resident['Total'], yTotal_pred)
    # print(predictedData)

    predict_US = predictedData['Trips by United States residents'].to_numpy()
    predict_CA = predictedData['Trips by Canadian residents'].to_numpy()
    predict_OTHER = predictedData['Trips by all other countries residents'].to_numpy()
    predictstr = predictedData['dates'].to_numpy()

    fig3, ax3 = plt.subplots(figsize=(10, 10))

    ax3.plot(predictstr, predict_CA, '-r', label='Canadian Residents')
    ax3.plot(predictstr, predict_US, '-b', label='United States Residents')
    ax3.plot(predictstr, predict_OTHER, '-g', label='Other Country Residents')
    plt.ticklabel_format(style='plain', axis='y')
    plt.xticks(rotation=75)
    plt.yticks(np.arange(0, 6200000, 200000))
    plt.legend()
    plt.savefig('predict.png')

    predictedDataWC = pd.DataFrame(columns=['dates', 'Trips by Canadian residents', 'Trips by United States residents',
                                            'Trips by all other countries residents', 'Total'])
    predictedDataWC['dates'] = np.append(datestr_after_covid, datestr_predict)
    predictedDataWC['Trips by Canadian residents'] = np.append(
        travel_residentOnlyAfterCovid['Trips by Canadian residents'], yCanWC_pred)
    predictedDataWC['Trips by United States residents'] = np.append(
        travel_residentOnlyAfterCovid['Trips by United States residents'], yUSWC_pred)
    predictedDataWC['Trips by all other countries residents'] = np.append(
        travel_residentOnlyAfterCovid['Trips by all other countries residents'], yOtherWC_pred)
    predictedDataWC['Total'] = np.append(travel_residentOnlyAfterCovid['Total'], yTotalWC_pred)
    # print(predictedDataWC)

    # Graphing by resident

    predict_US_WC = predictedDataWC['Trips by United States residents'].to_numpy()
    predict_CA_WC = predictedDataWC['Trips by Canadian residents'].to_numpy()
    predict_OTHER_WC = predictedDataWC['Trips by all other countries residents'].to_numpy()
    predictstr_WC = predictedDataWC['dates'].to_numpy()

    fig4, ax4 = plt.subplots(figsize=(10, 10))

    ax4.plot(predictstr_WC, predict_CA_WC, '-r', label='Canadian Residents')
    ax4.plot(predictstr_WC, predict_US_WC, '-b', label='United States Residents')
    ax4.plot(predictstr_WC, predict_OTHER_WC, '-g', label='Other Country Residents')
    plt.xticks(rotation=75)
    plt.yticks(np.arange(0, 900000, 50000))
    plt.legend()
    plt.savefig('predict_WC.png')

    print("pvalue for stats with pre covid data \n")
    print(stats.normaltest(predictedData['Trips by Canadian residents']).pvalue)
    print(stats.normaltest(predictedData['Trips by United States residents']).pvalue)
    print(stats.normaltest(predictedData['Trips by all other countries residents']).pvalue)
    print(stats.normaltest(predictedData['Total']).pvalue)

    print("pvalue for stats without pre covid data \n")
    print(stats.normaltest(predictedDataWC['Trips by Canadian residents']).pvalue)
    print(stats.normaltest(predictedDataWC['Trips by United States residents']).pvalue)
    print(stats.normaltest(predictedDataWC['Trips by all other countries residents']).pvalue)
    print(stats.normaltest(predictedDataWC['Total']).pvalue)


if __name__ == '__main__':
    main()
