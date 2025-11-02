import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('world_sustainability_dataset.csv')   # adjust filename/path

print(df.shape)
print(df.head())
print(df.info())
print(df.describe())

target = 'Sustainability_Score'  
features = [col for col in df.columns if col not in ['Country','Year', target]]

X = df[features]
y = df[target]

X = X.fillna(X.mean())
y = y.fillna(y.mean())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE on test set: {mse:.4f}')
print(f'R² on test set: {r2:.4f}')

importances = model.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
print('Top 10 important features:')
print(feat_imp.head(10))

plt.figure(figsize=(10,6))
sns.barplot(x=feat_imp.head(10), y=feat_imp.head(10).index)
plt.title('Top10 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
print('Cross-validation R² scores:', cv_scores)
print('Mean CV R²:', np.mean(cv_scores))
