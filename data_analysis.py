'''
    data_analysis.py
    Joseph Burson, 6/25/2016

    Analysis of Northern Ireland voting results in the 2016 EU membership referendum.
'''

import pandas
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

def main():
    df = pandas.read_csv("eu_ref_results_ni.csv", na_values="*", index_col=0)
    income_df = pandas.read_csv("ni_income.csv", na_values="*", index_col=0)
    religion_df = pandas.read_csv("ni_rel.csv", na_values="*", index_col=0)

    df['PercentCatholic'] = 100 * religion_df['Catholic'] / religion_df['Total']
    df['PercentProtestant'] = 100 * religion_df['Protestant'] / religion_df['Total']
    df['PercentRemain'] = 100 * df['Remain'] / df['Total']
    df['PercentLeave'] = 100 * df['Leave'] / df['Total']
    df['Income'] = income_df['All Persons Mean Wage']
    df['LeaveBool'] = df['Leave'] > df['Remain']
    df['PluralityCathBool'] = religion_df['Catholic'] > religion_df['Protestant']
    df['diffPerCathProt'] = df['PercentCatholic']-df['PercentProtestant']

    #print(df)

    # Crosstab analysis of whether a constituency is Catholic v. whether it
    #   voted to leave.
    xt1 = pandas.crosstab(df['PluralityCathBool'],df['LeaveBool'])
    #print(xt1)

    # Linear regression (fitted using OLS) of percent Catholic v. percent voting
    #   to leave
    lm1 = smf.ols(formula='PercentLeave ~ PercentCatholic', data=df)
    res1 = lm1.fit()
    print(res1.summary())

    # Scatterplot of percent Catholic v. percent voting to leave
    plot = df.plot(kind="scatter", x="PercentCatholic", y="PercentLeave")
    X_plot = np.linspace(10,90,100)
    plt.plot(X_plot, X_plot*res1.params[1] + res1.params[0])
    plt.show()


main()
