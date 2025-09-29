import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error
from sqlalchemy import create_engine
import pywt

# 1. Connect to PostgreSQL
DB_NAME = "weatherdatabasesmallextra"
DB_USER = "postgres"
DB_PASSWORD = "1314"
DB_HOST = "localhost"
DB_PORT = "5432"

engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# 2. Fetch data
df = pd.read_sql("SELECT * FROM weather_features ORDER BY datetime;", engine)

# Remove meta data
df = df.drop(columns=["id", "datetime"], errors="ignore")

# Define targets
target_columns = [
    #"rain",
    #"rain_ma6",
    #"rain_minutes",
    "average_temperature",
    "maximum_temperature",
    "minimum_temperature",
    "average_windspeed",
    "maximum_windspeed",
    "pressure",
    "cloud",
    "humidity",
    "sun",
    "wind_dir"
]

# --- FFT & DWT helpers ---
def extract_fft_features(series, window_size=48, n_freqs=5):
    feats = []
    for i in range(window_size, len(series)):
        w = series[i-window_size:i]
        vals = np.abs(np.fft.fft(w))[:n_freqs]
        feats.append(vals)
    return np.array(feats)

def extract_dwt_features(series, wavelet='db4', level=1, coeffs_to_keep=5):
    feats = []
    for i in range(level*coeffs_to_keep, len(series)):
        w = series[i-level*coeffs_to_keep:i]
        coeffs = pywt.wavedec(w, wavelet=wavelet, level=level)
        flat = np.hstack([c[:coeffs_to_keep] for c in coeffs])
        feats.append(flat)
    return np.array(feats)

# --- Build FFT/DWT features for selected series ---
series_map = {
    #'rain_ma6': df['rain_ma6'].values,
    'average_temperature': df['average_temperature'].values,
    'pressure': df['pressure'].values,
    'humidity': df['humidity'].values,
    'wind_dir': df['wind_dir'].values,
    'average_windspeed': df['average_windspeed'].values
}

fft_list, dwt_list = [], []
for name, arr in series_map.items():
    fft_list.append(extract_fft_features(arr))
    dwt_list.append(extract_dwt_features(arr))

#%%
# Align lengths
min_len = min(len(df), min(m.shape[0] for m in fft_list), min(m.shape[0] for m in dwt_list))

fft_all = np.hstack([m[-min_len:] for m in fft_list])
dwt_all = np.hstack([m[-min_len:] for m in dwt_list])

# --- Interaction features ---
df["pressure_x_humidity"] = df["pressure"] * df["humidity"]
df["temp_x_humidity"] = df["average_temperature"] * df["humidity"]
df["wind_x_cloud"] = df["average_windspeed"] * df["cloud"]

# Features
feature_columns = [col for col in df.columns if col not in target_columns]

# Build arrays
X_sql = df[feature_columns].astype(float).values[-min_len:]
y_all = df[target_columns].astype(float).values[-min_len:]

X_all = np.concatenate([X_sql, fft_all, dwt_all], axis=1)

# --- Scaling ---
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_all)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y_all)

# --- PCA ---
pca = PCA(n_components=0.95)  # keep 95% of variance
X_scaled = pca.fit_transform(X_scaled)
print("Antal features efter PCA:", X_scaled.shape[1])

# --- Split train/val/test ---
train_size = int(len(X_scaled) * 0.7)
valid_size = int(len(X_scaled) * 0.85)

X_train, y_train = X_scaled[:train_size], y_scaled[:train_size]
X_valid, y_valid = X_scaled[train_size:valid_size], y_scaled[train_size:valid_size]
X_test, y_test   = X_scaled[valid_size:], y_scaled[valid_size:]

# 6. Lav sekvenser
def create_sequences(X, y, timesteps):
    X_seq = np.array([X[i:i+timesteps] for i in range(len(X)-timesteps)])
    y_seq = np.array([y[i+timesteps] for i in range(len(y)-timesteps)])
    return X_seq, y_seq

timesteps = 20
X_train, y_train = create_sequences(X_train, y_train, timesteps)
X_valid, y_valid = create_sequences(X_valid, y_valid, timesteps)
X_test, y_test   = create_sequences(X_test, y_test, timesteps)

#%%
# 7. Byg LSTM-model
n_features = X_train.shape[2]
n_targets = y_train.shape[1]

model = Sequential([
    LSTM(256, return_sequences=True, kernel_regularizer=l2(0.01), input_shape=(timesteps, n_features)),
    BatchNormalization(),
    Dropout(0.2),

    LSTM(128, return_sequences=True),
    BatchNormalization(),
    Dropout(0.2),

    LSTM(64),
    BatchNormalization(),
    Dropout(0.2),

    Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.2),
    
    Flatten(),
    Dense(n_targets)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 8. Træn model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_valid, y_valid),
    callbacks=[early_stopping]
)

#%%
# 9. Evaluer model
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, MAE: {mae:.4f}")

y_pred = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\nModel Performance Metrics:")
print(f"R² Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

print("\nR²-score pr. target:")
for i, name in enumerate(target_columns):
    r2_i = r2_score(y_test[:, i], y_pred[:, i])
    print(f"{name:>25}: R² = {r2_i:.4f}")

model.save('C:/Users/shans/OneDrive/Skrivebord/VejrAI/VejrAILSTM.h5')
