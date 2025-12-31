import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

df = pd.read_csv('car_price_prediction.csv')

df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(r'[^\w]+', '_', regex=True)
)


df.drop(columns=['id'], inplace=True)

#standardize missing values
missing_values = ['-', '—', ' ', '', 'None', 'null']
df.replace(missing_values, np.nan, inplace=True)

df.dropna(subset=['levy'], inplace=True)

#create is_turbo column 
df['is_turbo'] = (
    df['engine_volume']
    .astype(str)
    .str.contains(r'turbo|\bt\b', case=False, regex=True)
    .astype(int)
)

#extract numeric engine volume ONLY
df['engine_volume'] = (
    df['engine_volume']
    .astype(str)
    .str.replace(r'[^0-9\.]', '', regex=True)
)

df['engine_volume'] = pd.to_numeric(df['engine_volume'], errors='coerce')


df['mileage'] = (
    df['mileage']
    .astype(str)                    
    .str.replace(r'[^0-9]', '', regex=True)  #used to remove km
    .astype(float)  
)


x = df.drop('price', axis=1)
y = df['price']


numerical_cols = [
    'levy', 'prod_year', 'cylinders',
    'airbags', 'engine_volume','mileage', 'is_turbo'
]

categorical_cols = [
    'manufacturer', 'model', 'category',
    'gear_box_type', 'drive_wheels',
    'doors', 'wheel', 'color', 'leather_interior', 'fuel_type',
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


joblib.dump(model, "car_price_prediction.pkl")


#get feature importances

feature_names = model.named_steps['preprocessor'].get_feature_names_out()

importances = model.named_steps['regressor'].feature_importances_

fi = pd.DataFrame({
    'feature':feature_names,
    'importance':importances
}).sort_values(by='importance', ascending=False)



