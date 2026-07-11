import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset.zip")
#print(df.head())
#print(df.columns)

#print(df.isnull().sum())
# print(df.duplicated().sum())
# print(df.dtypes)
#all features are in  either float or int datatype

# for col in df.columns:
#     print(col)
#     print(df[col].unique()[:10])
#     print("-------------------------")

#print target values
#print(df["generated_power_kw"].head() )

# plt.figure(figsize=(8,5))
# sns.histplot(df["generated_power_kw"], kde=True)

# plt.title("Distribution of Generated Solar Power")
# plt.xlabel("Generated Power (kW)")
# plt.ylabel("Frequency")
# plt.show()

selected_features = [
    "angle_of_incidence",
    "total_cloud_cover_sfc",
    "zenith",
    "azimuth",
    "shortwave_radiation_backwards_sfc",
    "total_precipitation_sfc",
    "low_cloud_cover_low_cld_lay",
    "wind_gust_10_m_above_gnd"
]

X = df[selected_features]
y = df["generated_power_kw"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
# print("Training Data :", X_train.shape)
# print("Testing Data :", X_test.shape)

# from sklearn.linear_model import LinearRegression  #  does not gives better result than Decision Tree
# lr =  LinearRegression()
# lr.fit(X_train,y_train)
# y_pred = lr.predict(X_test)

from xgboost import XGBRegressor  #   gives better result than Decision Tree
xgb = XGBRegressor(random_state = 42)
xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)

# from sklearn.tree import DecisionTreeRegressor  #  does not gives better result than other
# dt = DecisionTreeRegressor()
# dt.fit(X_train,y_train)
# y_pred = dt.predict(X_test)

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

# print("mean_absolute_error",mae)
# print("mean_squared_error",mse)
# print("root_mean_squared_error",rmse)
# print("r2_score",mae)

results = pd.DataFrame({"Actual": y_test,"Predicted": y_pred})
results.head()

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
#plt.show()

# print("Training Score :", xgb.score(X_train, y_train))  #0.9905
# print("Testing Score :", xgb.score(X_test, y_test))  #0.8197

from sklearn.model_selection import RandomizedSearchCV
param_grid = {'n_estimators': [100, 200, 300],'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]}
random_search = RandomizedSearchCV(estimator=xgb,param_distributions=param_grid,
 n_iter=20,scoring='r2',cv=5,verbose=2,random_state=42,n_jobs=-1)

random_search.fit(X_train, y_train)
# print(random_search.best_params_)

best_xgb = random_search.best_estimator_
y_pred = best_xgb.predict(X_test)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_xgb.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

#print(importance)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# print("MAE :", mae)
# print("RMSE :", rmse)
# print("R² :", r2)

import joblib
import os
os.makedirs("model", exist_ok=True)
joblib.dump(best_xgb, "model/xgboost_model.pkl")

feature_names = X.columns.tolist()
joblib.dump(feature_names,"model/feature_names.pkl")


