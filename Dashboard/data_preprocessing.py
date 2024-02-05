import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

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

    if dataset == 'titanic':
        # Ensure that the dataset can be accessed from various scripts
        current_directory = os.path.dirname(os.path.abspath(__file__))
        train_csv_path = os.path.join(current_directory, '..', 'Titanic', 'train.csv')
        train_data_raw = pd.read_csv(train_csv_path)
        X = train_data_raw.drop('Survived', axis=1)
        y = train_data_raw['Survived']

        X = X.dropna(subset=['Embarked', 'Age'])
        # Update y to keep only the rows that are still in X
        y = y.loc[X.index]

        #
        X['Pclass'] = X['Pclass'].astype('object')
        X = X.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
        X['Members'] = X['SibSp'] + X['Parch']
        X = X.drop(columns=['SibSp', 'Parch'])

        exclude = ['Pclass', 'Sex', 'Embarked']
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
