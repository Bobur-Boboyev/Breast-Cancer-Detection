import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

def preprocessing_data(path):
    data = pd.read_csv(path)
    data_cleared = data.drop("Unnamed: 32", axis=1)
    data_cleared = data_cleared.drop("id", axis=1)
    
    x = data_cleared.drop("diagnosis", axis=1)
    y = data_cleared['diagnosis']

    scaler = StandardScaler()
    scaled_x = scaler.fit_transform(x)

    le = LabelEncoder()
    encoded_y = le.fit_transform(y)
    joblib.dump(scaler, "models/scaler.pkl")
    return train_test_split(scaled_x, encoded_y, test_size=0.2, random_state=42)