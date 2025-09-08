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
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.inspection import PartialDependenceDisplay
import shap

# De kolonner fra weather_data du vil lave FFT/DWT på
target_cols = ['rain','rain_minutes','average_temperature','maximum_temperature',
        'minimum_temperature','average_windspeed','maximum_windspeed',
        'pressure','cloud','humidity','sun']
# Og tilsvarende deres indeks i weather_data-arrayet
col_indices = [0, 9, 7]

# 1) DEFINITION AF HJÆLPEFUNKTIONER
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

def print_r2(epoch, logs):
    if epoch % 5 == 0 or epoch == 0:
        y_val_pred = model.predict(X_valid, verbose=0)
        print(f"\n Epoch {epoch+1}: R² på validering:")
        for i, name in enumerate(target_names):
            r2v = r2_score(y_valid[:, i], y_val_pred[:, i])
            print(f"{name:>25}: R² = {r2v:.4f}")

def extract_fft_features(series, window_size=48, n_freqs=5):
    feats = []
    for i in range(window_size, len(series)):
        w = series[i-window_size:i]
        vals = np.abs(np.fft.fft(w))[:n_freqs]
        feats.append(vals)
    return np.array(feats)

def extract_dwt_features(series, wavelet='db4', level=2, coeffs_to_keep=5):
    feats = []
    for i in range(level*coeffs_to_keep, len(series)):
        w = series[i-level*coeffs_to_keep:i]
        coeffs = pywt.wavedec(w, wavelet=wavelet, level=level)
        flat = np.hstack([c[:coeffs_to_keep] for c in coeffs])
        feats.append(flat)
    return np.array(feats)

# 2) INDLÆSNING & FFT/DWT FEATURES
df_raw = pd.read_excel("C:/Users/shans/OneDrive/Skrivebord/VejrAI/NewDatasetsmallENG.xlsx",
                       engine="openpyxl", header=None)
df_raw = df_raw.iloc[1:].reset_index(drop=True)
dates = pd.to_datetime(df_raw.iloc[:,0])
weather = df_raw.iloc[:,1:12].astype(float).values

time_feats = np.column_stack([
    np.sin(2*np.pi*dates.dt.month/12),
    np.cos(2*np.pi*dates.dt.month/12),
    dates.dt.day/31,
    dates.dt.hour/24
])

cols = ['rain','rain_minutes','average_temperature','maximum_temperature',
        'minimum_temperature','average_windspeed','maximum_windspeed',
        'pressure','cloud','humidity','sun']
idxs = [0,1,2,3,4,5,6,7,8,9,10]

fft_list, dwt_list = [], []

for col, idx in zip(target_cols, col_indices):
    ser = weather[:,idx]
    fft_list.append(extract_fft_features(ser))
    dwt_list.append(extract_dwt_features(ser))

min_len = min(len(a) for a in fft_list+dwt_list)
weather = weather[-min_len:]
time_feats = time_feats[-min_len:]
dates = dates[-min_len:]
fft_all = np.hstack([a[-min_len:] for a in fft_list])
dwt_all = np.hstack([d[-min_len:] for d in dwt_list])

X_all = np.concatenate([weather, time_feats, fft_all, dwt_all],axis=1)
y_all = weather

scaler_X, scaler_y = StandardScaler(), MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_all)
y_scaled = scaler_y.fit_transform(y_all)

# Split tidsserie
n = len(X_scaled)
t1, t2 = int(n*0.7), int(n*0.85)
X_tr, X_va, X_te = X_scaled[:t1], X_scaled[t1:t2], X_scaled[t2:]
y_tr, y_va, y_te = y_scaled[:t1], y_scaled[t1:t2], y_scaled[t2:]

# Sekvenser
timesteps = 48
def seq(X,y):
    return (np.array([X[i:i+timesteps] for i in range(len(X)-timesteps)]),
            np.array([y[i+timesteps] for i in range(len(y)-timesteps)]))
X_train, y_train = seq(X_tr,y_tr)
X_valid, y_valid = seq(X_va,y_va)
X_test,  y_test  = seq(X_te,y_te)

print("Train shape:",X_train.shape)

#%%
# 3) MODEL: Transformer + R²-logger
input_layer = Input(shape=(timesteps, X_train.shape[2]))
proj = Dense(256,activation='relu')(input_layer)
x = transformer_block(proj,4,32,256,0.3)
x = transformer_block(x,6,32,384,0.3)
x = transformer_block(x,8,32,512,0.3)
x = Flatten()(x)
x = Dense(512,activation='relu',kernel_regularizer=l2(0.001))(x)
x = Dropout(0.3)(x)
x = Dense(128,activation='relu',kernel_regularizer=l2(0.001))(x)
x = Dropout(0.3)(x)
output_layer = Dense(11)(x)
model = Model(input_layer, output_layer)
model.compile('adam', loss='mse', metrics=['mae'])
target_names = cols

r2_logger = LambdaCallback(on_epoch_end=print_r2)

model.fit(X_train,y_train,epochs=1,batch_size=32,
          validation_data=(X_valid,y_valid), callbacks=[r2_logger])

loss, mae = model.evaluate(X_test,y_test)
print("Test MSE:",loss,"MAE:",mae)
y_pred = model.predict(X_test)
print("R² samlet:",r2_score(y_test,y_pred))

print("\nR² per klasse:")
for i,nm in enumerate(target_names):
    print(f"{nm:>20}: {r2_score(y_test[:,i],y_pred[:,i]):.4f}")
#%%
# 4) EDA: PCA + KMeans
df_feats = pd.DataFrame(X_all, columns=[f"f{i}" for i in range(X_all.shape[1])])
df_feats['rain_total'] = y_all[:, 0]

# Korrelations-heatmap
corr = df_feats.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.show()

# PCA: fit én gang og genbrug objektet
pca = PCA(n_components=2)
Xp = pca.fit_transform(df_feats.drop(columns=['rain_total']))
print("PCA varians:", pca.explained_variance_ratio_)

# KMeans clustering på PCA-komponenterne
kmeans = KMeans(n_clusters=4, random_state=42).fit(Xp)
plt.scatter(Xp[:, 0], Xp[:, 1], c=kmeans.labels_)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('KMeans clustering (PCA)')
plt.show()

# 5) Klassifikation & kalibrering
df_cls = df_feats.copy()
df_cls['rain_flag'] = (df_cls['rain_total'] > 0).astype(int)
Xc, yc = df_cls.drop(columns=['rain_total', 'rain_flag']), df_cls['rain_flag']

split = int(len(Xc) * 0.85)
Xc_tr, Xc_te = Xc[:split], Xc[split:]
yc_tr, yc_te = yc[:split], yc[split:]

tscv = TimeSeriesSplit(5)
base = RandomForestClassifier(100, random_state=42)
cal = CalibratedClassifierCV(base, method='isotonic', cv=tscv).fit(Xc_tr, yc_tr)

probs = cal.predict_proba(Xc_te)[:, 1]
print("Brier:", brier_score_loss(yc_te, probs),
      "AUC:", roc_auc_score(yc_te, probs))

plt.plot(*calibration_curve(yc_te, probs, n_bins=10), marker='o')
plt.show()

#%%
# 6) SHAP & PDP - Brug en præ-trænet model til analyse
# Træn en separat RandomForest direkte på træningsdata
rf_for_shap = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
).fit(Xc_tr, yc_tr)

# SHAP-forklaringer
expl = shap.TreeExplainer(rf_for_shap)
sv = expl.shap_values(Xc_tr)

# Håndter binary/multiclass:
if isinstance(sv, list) and len(sv) > 1:
    shap_vals_class = sv[1]  # klasse 1 for classification
else:
    shap_vals_class = sv  # regression eller binært output

# Valider at dimensioner matcher
assert shap_vals_class.shape[1] == Xc_tr.shape[1], \
    f"Uoverensstemmelse: SHAP features={shap_vals_class.shape[1]}, data features={Xc_tr.shape[1]}"

# Barplot over gennemsnitlig absolut SHAP-værdi
shap.summary_plot(shap_vals_class, Xc_tr, plot_type='bar', feature_names=Xc_tr.columns)

# Partial Dependence Plots for top 3 features
top3_idx = np.argsort(np.abs(shap_vals_class).mean(0))[-3:]
for idx in top3_idx:
    PartialDependenceDisplay.from_estimator(
        rf_for_shap,
        Xc_tr,
        [Xc_tr.columns[idx]]
    )
    plt.show()

print("Job's done!")
