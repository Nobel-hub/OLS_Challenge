import warnings
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from data_processing import split_data

def correlation_among_features(df, col):
    numeric_cols = df[col]
    corr = numeric_cols.corr()
    corr_features = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > 0.8:  # Corrected condition
                colname = corr.columns[i]
                corr_features.add(colname)
    return corr_features


def lr_model(x_train, y_train):
    x_train_with_intercept = sm.add_constant(x_train)
    lr = sm.OLS(y_train, x_train_with_intercept).fit()
    return lr


def identify_significant_vars(lr, p_value=0.05):
    significant_vars = [var for var in lr.pvalues.keys() if lr.pvalues[var] < p_value]
    return significant_vars


if __name__ == '__main__':
    capped_data = pd.read_csv('Clean_data.csv')
    correlated_features = correlation_among_features(capped_data, capped_data.columns)
    # print(correlated_features)
    correlated_features_list = ['povertyPercent', 'State_ District of Columbia', 'PctEmpPrivCoverage',
                                'PctPrivateCoverage', 'PctPrivateCoverageAlone', 'PctMarriedHouseholds',
                                'upperbound', 'PctPublicCoverageAlone', 'lowerbound',
                                'County_Los Angeles County', 'MedianAgeFemale', 'median']
    cols = [col for col in capped_data.columns if col not in correlated_features_list]
    # print(len(cols))
    x_train, x_test, y_train, y_test = split_data(capped_data[cols], "TARGET_deathRate")
    lr = lr_model(x_train, y_train)
    summary = lr.summary()
    print(summary)
    significant_vars = identify_significant_vars(lr)
    x_train = sm.add_constant(x_train)
    lr = lr_model(x_train[significant_vars], y_train)
    summary = lr.summary()
    print(summary)

