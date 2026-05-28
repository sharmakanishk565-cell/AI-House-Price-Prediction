
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
data = pd.read_csv("dataset/Housing.csv")

# Convert yes/no values
binary_cols = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'prefarea'
]

for col in binary_cols:
    data[col] = data[col].map({
        'yes': 1,
        'no': 0
    })

# Convert furnishing status
data['furnishingstatus'] = data['furnishingstatus'].map({
    'furnished': 2,
    'semi-furnished': 1,
    'unfurnished': 0
})

# Features and target
X = data.drop('price', axis=1)
y = data['price']

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# Train AI model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Save files
joblib.dump(model, "house_price_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model trained successfully")

