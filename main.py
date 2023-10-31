import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from igann import IGANN
from sklearn.model_selection import train_test_split
import warnings

# Suppress all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)




X, y = load_diabetes(return_X_y=True, as_frame=True)
scaler = StandardScaler()
X_names = X.columns

X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=X_names)
X['sex'] = X.sex.apply(lambda x: 'w' if x > 0 else 'm')
y = (y - y.mean()) / y.std()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = IGANN(task='regression')
model.fit(X_train, y_train)

single_row_df = X_test.iloc[[0]]
print(model.predict(single_row_df))

model.plot_single(plot_by_list=['age', 'bmi', 'bp', 'sex', 's1', 's2'])
