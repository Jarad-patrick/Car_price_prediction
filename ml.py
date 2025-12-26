import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('car_price_prediction.csv')

df.drop(columns=['ID'], inplace=True)

#standardize missing values
missing_values = ['-', '—', ' ', '', 'None', 'null']
df.replace(missing_values, np.nan, inplace=True)

df.dropna(subset=['Levy'], inplace=True)

#create is_turbo column 
df['is_turbo'] = (
    df['Engine volume']
    .astype(str)
    .str.contains(r'turbo|\bt\b', case=False, regex=True)
    .astype(int)
)

#extract numeric engine volume ONLY
df['Engine volume'] = (
    df['Engine volume']
    .astype(str)
    .str.replace(r'[^0-9\.]', '', regex=True)
)

df['Engine volume'] = pd.to_numeric(df['Engine volume'], errors='coerce')


df['Mileage'] = (
    df['Mileage']
    .astype(str)                    
    .str.replace(r'[^0-9]', '', regex=True)  #used to remove km
    .astype(float)  
)


x = df.drop('Price', axis=1)
y = df['Price']


numerical_cols = [
    'Levy', 'Prod. year', 'Cylinders',
    'Airbags', 'Engine volume','Mileage', 'is_turbo'
]

categorical_cols = [
    'Manufacturer', 'Model', 'Category',
    'Gear box type', 'Drive wheels',
    'Doors', 'Wheel', 'Color', 'Leather interior', 'Fuel type',
]

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size= 0.2,
    random_state=42
)

preprocessor  = ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),numerical_cols),
        ('cat',OneHotEncoder(handle_unknown='ignore'),categorical_cols)
    ]
)

model = Pipeline(
    steps=[
        ('preprocessor',preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test,y_pred)
rmse = np.sqrt (mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test, y_pred)


print(f"MAE: {mae:,.0f}")
print(f"RMSE: {rmse:,.0f}")
print(f"R²: {r2:.3f}")



#get feature importances 

feature_names = model.named_steps['preprocessor'].get_feature_names_out()

importances = model.named_steps['regressor'].feature_importances_

fi = pd.DataFrame({
    'feature':feature_names,
    'importance':importances
}).sort_values(by='importance', ascending=False)

print(fi.head(20))