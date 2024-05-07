import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def bin_to_num(data):
    binnedinc = []

    for i in data['binnedInc']:
        i = i.strip("()[]")
        i = i.split(",")
        i = tuple(i)
        i = tuple(map(float, i))
        i = list(i)
        binnedinc.append(i)
    data['binnedInc'] = binnedinc
    data['lowerbound'] = [i[0] for i in data["binnedInc"]]
    data['upperbound'] = [i[1] for i in data["binnedInc"]]
    data['median'] = (data['lowerbound'] + data['upperbound'])/2
    data.drop("binnedInc", axis=1, inplace=True)
    return data

def cat_to_col(data):
    data['County'] = [i.split(',')[0] for i in data["Geography"]]
    data['State'] = [i.split(',')[1] for i in data["Geography"]]
    data.drop("Geography", axis=1, inplace=True)
    return data


def one_hot_encoding(data):
    categorical_columns = data.select_dtypes(include=["object"]).columns
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    one_hot_encoded = one_hot_encoder.fit_transform(data[categorical_columns])
    one_hot_encoded = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(categorical_columns))
    data = data.drop(categorical_columns, axis=1)
    data = pd.concat([data, one_hot_encoded], axis=1)
    return data
