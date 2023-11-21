import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    X['sex'] = X.sex.apply(lambda x: 'w' if x > 0 else 'm')
    y = (y - y.mean()) / y.std()
    return train_test_split(X, y, test_size=0.2, random_state=42)