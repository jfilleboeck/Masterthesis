import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
from ucimlrepo import fetch_ucirepo


def load_and_preprocess_data(dataset='diabetes'):
    if dataset == 'diabetes':
        X, y = load_diabetes(return_X_y=True, as_frame=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)
        X['sex'] = X.sex.apply(lambda x: 'w' if x > 0 else 'm')
        y = (y - y.mean()) / y.std()
        task = 'regression'
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test, task


    if dataset == 'bike':
        bike_sharing_dataset = fetch_ucirepo(id=275)

        # data (as pandas dataframes)
        X = bike_sharing_dataset.data.features
        y = bike_sharing_dataset.data.targets
        X.dropna()
        X = X.drop(columns=['dteday'])

        # Update y to keep only the rows that are still in X
        y = y.loc[X.index]
        y = (y - y.mean()) / y.std()
        numerical_features = ['temp', 'atemp', 'hum', 'windspeed']

        X_numeric = X[numerical_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_numeric)
        X_scaled = pd.DataFrame(X_scaled, columns=X_numeric.columns)
        X_scaled = pd.DataFrame(X_scaled, columns=X_numeric.columns,
                                index=X_numeric.index)  # Preserve the original index

        categorical_features = [feature for feature in X if feature not in numerical_features]
        X_categorical = X.loc[:, categorical_features].astype('object')


        X = pd.concat([X_scaled, X_categorical], axis=1)
        #X = X.astype('object')

        #X = X[['hr', 'mnth', 'weekday']].astype('object')


        X_train, X_test, y_train, y_test = train_test_split(X, np.array(y), test_size=0.2, random_state=42)
        task = "regression"

        return X_train, X_test, y_train, y_test, task

    if dataset == 'adult':
        adult = fetch_openml(name='adult', version=2)
        X, y = adult.data, adult.target

        # Remove rows with empty values
        X.dropna()
        # Update y to keep only the rows that are still in X
        y = y.loc[X.index]

        #X = X.drop(columns=['instant', 'dteday'])


        numerical_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        X_numeric = X[numerical_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_numeric)
        X_scaled = pd.DataFrame(X_scaled, columns=X_numeric.columns)
        X_scaled = pd.DataFrame(X_scaled, columns=X_numeric.columns,
                                index=X_numeric.index)

        categorical_features = [feature for feature in adult.feature_names if feature not in numerical_features]
        X_categorical = X.loc[:, categorical_features].astype('object')

        # Convert the datatypes of categorical features to 'object'
        #X_categorical = X_categorical.astype('object')

        X = pd.concat([X_scaled, X_categorical], axis=1)

        y_mapped = y.map({'>50K': 1, '<=50K': 0})

        X_train, X_test, y_train, y_test = train_test_split(X, np.array(y_mapped), test_size=0.2, random_state=42)

        task = "classification"


        return X_train, X_test, y_train, y_test, task


    if dataset == 'iris':
        iris = load_iris()
        X, y = iris.data, iris.target
        # Filter the dataset to include only Iris-setosa and Iris-versicolor
        mask = (y == 0) | (y == 1)
        X_binary = X[mask]
        y_binary = y[mask]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_binary)
        X = pd.DataFrame(X_scaled, columns=iris.feature_names)
        task = 'classification'
        X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test, task


    if dataset == 'titanic':
        # Ensure that the dataset can be accessed from various scripts
        current_directory = os.path.dirname(os.path.abspath(__file__))
        train_csv_path = os.path.join(current_directory, '..', 'Titanic', 'train.csv')
        train_data_raw = pd.read_csv(train_csv_path)
        X = train_data_raw.drop('Survived', axis=1)
        y = train_data_raw['Survived']
        # Change X to lower case
        X.columns = [col[0].lower() + col[1:] if col else col for col in X.columns]
        X = X.dropna(subset=['embarked', 'age'])

        # Update y to keep only the rows that are still in X
        y = y.loc[X.index]
        #
        X['pclass'] = X['pclass'].astype('object')
        X = X.drop(columns=['passengerId', 'name', 'ticket', 'cabin'])
        X['members'] = (X['sibSp'] + X['parch'])
        #X['Members'] = X['Members'].astype('object')
        X = X.drop(columns=['sibSp', 'parch'])

        exclude = ['pclass', 'sex', 'embarked', 'members']
        X_numeric = X.drop(columns=exclude)
        X_categorical = X[exclude]

        # Apply StandardScaler to numeric columns
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_numeric)
        X_scaled = pd.DataFrame(X_scaled, columns=X_numeric.columns)

        # Reset indices before concatenation
        X_scaled = X_scaled.reset_index(drop=True)
        X_categorical = X_categorical.reset_index(drop=True)

        # Concatenate the categorical and numerical features
        X = pd.concat([X_scaled, X_categorical], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        task = "classification"
        return X_train, X_test, y_train, y_test, task


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, task = load_and_preprocess_data("titanic")
