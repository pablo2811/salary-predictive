#!/usr/bin/env python3

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

def pre_process(df):
    df[df == "?"] = np.nan
    df.dropna(inplace = True)
    data = df.drop(["fnlwgt", "income_level"], axis=1)
    target = np.array(df["income_level"])
    return data,target

def get_split(df,test_size_fraction):
    data, target = pre_process(df)
    X_train, X_test, y_train, y_test = train_test_split(data, target,
                                                        test_size = test_size_fraction,
                                                        random_state = 42,
                                                       stratify=target)
    return X_train, X_test, y_train, y_test

def grouping_marital(df):
    res = df.copy()
    res['marital_status'] = res['marital_status'].replace(
        ['Widowed', 'Divorced', 'Separated', 'Never-married', 'Married-spouse-absent'], 'Living-Alone')
    res['marital_status'] = res['marital_status'].replace(
        ['Married-civ-spouse', 'Married-AF-spouse'], 'Married')
    return res
    
def get_best_params(model_type, param_test, param_grid, df):
    test_model = model_type(**param_test)
    pipeline = get_pipeline(test_model)
    randomizer = RandomizedSearchCV(pipeline, param_grid, cv=3, n_iter=5)
    X_train, X_test, y_train, y_test = get_split(grouping_marital(df),0.2)
    randomizer.fit(X_train,y_train)
    return randomizer.best_params_


def parse_params(best_params):
    parsed_dicitionary_params = dict()
    for k in best_params.keys():
        parsed_dicitionary_params[k.replace("model__","")] = best_params[k]
    parsed_dicitionary_params["random_state"] = 42
    parsed_dicitionary_params["n_jobs"] = -1
    return parsed_dicitionary_params

def result_tuned_model(df,model_type, param_test, param_grid, name):
    best_params = parse_params(get_best_params(model_type, param_test, param_grid, df))
    designated_model = model_type(**best_params)
    transformed_data = grouping_marital(df)
    test_model(designated_model,transformed_data,name)

def test_model(model,data,label=None):
    if label is None:
        label = ""
    X_train,X_test,y_train,y_test = get_split(data,0.2)
    kfolds = 8
    split = KFold(n_splits=kfolds, shuffle=True, random_state=42)
    cv_results = cross_val_score(get_pipeline(model), 
                     X_train.drop("education_num", axis=1), y_train, 
                     cv=split,
                     scoring="accuracy",
                     n_jobs=-1)
    
    print(f" {label} cross validation accuarcy score: {round(np.mean(cv_results), 4)} +/- {round(np.std(cv_results), 4)} (std) \t min: {round(min(cv_results), 4)}, max: {round(max(cv_results), 4)}")
    
def get_pipeline(model):
    cat_features = ["workclass", "education", "marital_status",
                    "occupation", "relationship", "race", 
                    "sex", "native_country"]     

    transformer = ColumnTransformer(
        [
        ("onehot", OneHotEncoder(handle_unknown = 'ignore'), cat_features), 
        ("std_scaler", StandardScaler(), ["age", "capital_gain", "capital_loss", "hours_per_week"])
        ],
        remainder = "passthrough"
    )

    model_pipeline = Pipeline(
        [
            ('transformer', transformer),
            ('model', model)
        ]
    )
    return model_pipeline


def main():
	print("Running...")
	df = pd.read_csv("https://lovespreadsheet-tutorials.s3.amazonaws.com/APIDatasets/census_income_dataset.csv")
	param_test_rf = {
                "random_state": 42,
                 "n_jobs": -1
                }
	param_grid_rf = {
            "model__max_depth": [3, None],
            "model__max_features": [1, 3, 10],
            "model__min_samples_leaf": [1, 3, 10],
            "model__bootstrap": [True, False],
            "model__criterion": ["gini", "entropy"]
                }
	result_tuned_model(df,RandomForestClassifier,param_test_rf,param_grid_rf,"RandomForestClassifier")

if __name__ == "__main__":
	main()