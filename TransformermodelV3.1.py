#This version is a test of how the code performs without the features and data processing from the PostgreSQL database
import numpy as np
import pandas as pd
import pywt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Flatten, Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import LambdaCallback

def print_r2(epoch, logs):
    if epoch % 5 == 0 or epoch == 0:
        y_val_pred = model.predict(X_valid, verbose=0)
        print(f"Epoch {epoch+1}: R²-scorer på valideringsdata:")
        for i, name in enumerate(target_names):
            r2 = r2_score(y_valid[:, i], y_val_pred[:, i])
            print(f"{name:>25}: R² = {r2:.4f}")


# --- Transformer Block ---
def transformer_block(inputs, num_heads, key_dim, ff_dim, dropout_rate):
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
    attention = Dropout(dropout_rate)(attention)
    attention = Add()([inputs, attention])
    attention = LayerNormalization()(attention)

    ff_layer = Dense(ff_dim, activation="relu")(attention)
    ff_layer = Dropout(dropout_rate)(ff_layer)
    ff_layer = Dense(inputs.shape[-1])(ff_layer)
    ff_layer = Add()([attention, ff_layer])
    return LayerNormalization()(ff_layer)

# --- Load dataset ---
df = pd.read_excel("C:/Users/shans/OneDrive/Skrivebord/VejrAI/NewDatasetsmallENG.xlsx", engine="openpyxl", header=None) # issue. Since my pc uses the danish method for writing number. Using ',' not '.'. This caused great difficulties using a .csv file separated by both ',' and ';' for some reason.
df = df.iloc[1:, :]
dates = pd.to_datetime(df.iloc[:, 0])
weather_data = df.iloc[:, 1:12].astype(float).values  # 11 klasser

# --- Time features ---
time_features = np.column_stack([
    np.sin(2 * np.pi * dates.dt.month / 12),
    np.cos(2 * np.pi * dates.dt.month / 12),
    dates.dt.day / 31,
    dates.dt.hour / 24
])

# --- FFT + DWT Feature Extraction ---
def extract_fft_features(series, window_size=48, n_freqs=5):
    fft_feats = []
    for i in range(window_size, len(series)):
        window = series[i - window_size:i]
        fft_vals = np.fft.fft(window)
        fft_abs = np.abs(fft_vals)[:n_freqs]
        fft_feats.append(fft_abs)
    return np.array(fft_feats)

def extract_dwt_features(series, wavelet='db4', level=2, coeffs_to_keep=5):
    dwt_feats = []
    for i in range(level * coeffs_to_keep, len(series)):
        window = series[i - level * coeffs_to_keep:i]
        coeffs = pywt.wavedec(window, wavelet=wavelet, level=level)
        coeffs_flat = np.concatenate([c[:coeffs_to_keep] for c in coeffs])
        dwt_feats.append(coeffs_flat)
    return np.array(dwt_feats)

target_cols = ["rain", "humidity", "pressure"]
col_indices = [0, 9, 7]  # Indeks i weather_data
fft_list, dwt_list = [], []

for idx in col_indices:
    series = weather_data[:, idx]
    fft_feat = extract_fft_features(series)
    dwt_feat = extract_dwt_features(series)
    fft_list.append(fft_feat)
    dwt_list.append(dwt_feat)

# --- Synkroniser længde ---
min_len = min(len(f) for f in fft_list + dwt_list)
weather_data = weather_data[-min_len:]
time_features = time_features[-min_len:]
dates = dates[-min_len:]

fft_all = np.hstack([f[-min_len:] for f in fft_list])
dwt_all = np.hstack([d[-min_len:] for d in dwt_list])

# --- Kombiner alle input features ---
X_all = np.concatenate([weather_data, time_features, fft_all, dwt_all], axis=1)
y_all = weather_data  # Targets: alle 11 klasser

# --- Skaler input og output ---
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_all)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y_all)

# --- Tidsbaseret split ---
train_size = int(len(X_scaled) * 0.7)
valid_size = int(len(X_scaled) * 0.85)

X_train, y_train = X_scaled[:train_size], y_scaled[:train_size]
X_valid, y_valid = X_scaled[train_size:valid_size], y_scaled[train_size:valid_size]
X_test,  y_test  = X_scaled[valid_size:], y_scaled[valid_size:]

# --- Sekvenser ---
timesteps = 48
def create_sequences(X, y, t):
    X_seq = np.array([X[i:i+t] for i in range(len(X)-t)])
    y_seq = np.array([y[i+t] for i in range(len(y)-t)])
    return X_seq, y_seq

X_train, y_train = create_sequences(X_train, y_train, timesteps)
X_valid, y_valid = create_sequences(X_valid, y_valid, timesteps)
X_test,  y_test  = create_sequences(X_test,  y_test,  timesteps)

print(f"Træningssekvensers form: {X_train.shape}")
print(f"(batchs, timesteps, features) = {X_train.shape}")

#%%
# --- Model ---
input_layer = Input(shape=(timesteps, X_train.shape[2]))
projected = Dense(256, activation='relu')(input_layer)

x = transformer_block(projected, num_heads=4, key_dim=32,  ff_dim=256, dropout_rate=0.3)
x = transformer_block(x,         num_heads=6, key_dim=32,  ff_dim=384, dropout_rate=0.3)
x = transformer_block(x,         num_heads=8, key_dim=32,  ff_dim=512, dropout_rate=0.3)

x = Flatten()(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.3)(x)

output_layer = Dense(11)(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])

# --- Træning ---
r2_logger = LambdaCallback(on_epoch_end=print_r2)

model.fit(X_train, y_train,epochs=20, batch_size=32, validation_data=(X_valid, y_valid), callbacks=[r2_logger])

# --- Evaluering ---
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, MAE: {mae:.4f}")

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse_val = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse_val)

print(f"R² Score (samlet): {r2:.4f}")
print(f"MSE: {mse_val:.4f}")
print(f"RMSE: {rmse:.4f}")

# --- R² pr. klasse ---
target_names = [
    "rain", "rain_minutes", "average_temperature", "maximum_temperature",
    "minimum_temperature", "average_windspeed", "maximum_windspeed",
    "pressure", "cloud", "humidity", "sun"
]

print("\nR²-score pr. klasse:")
for i, name in enumerate(target_names):
    r2_individual = r2_score(y_test[:, i], y_pred[:, i])
    print(f"{name:>25}: R² = {r2_individual:.4f}")

# --- Gem model ---

model.save("C:/Users/shans/OneDrive/Skrivebord/VejrAI/VejrAI_Transformer_with_freq.h5")
