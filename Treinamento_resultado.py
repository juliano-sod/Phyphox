import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

#NO MAC USEI:
#pip3 install numpy pandas matplotlib seaborn scipy scikit-learn

TARGET_LENGTH = 200

# ---------------------------------------------------------
# 1. PRÉ-PROCESSAMENTO DE SÉRIES TEMPORAIS
# ---------------------------------------------------------
def preprocess_series(series, target_length=200, trim_percent=0.05):
    # Remoção das extremidades (ruído de início/fim de gravação)
    n = len(series)
    trim_start, trim_end = int(n * trim_percent), int(n * (1 - trim_percent))
    trimmed = series[trim_start:trim_end] if trim_end > trim_start else series
    
    # Reamostragem/Interpolação linear para padronização do vetor
    x_old = np.linspace(0, 1, len(trimmed))
    x_new = np.linspace(0, 1, target_length)
    f = interp1d(x_old, trimmed, kind='linear', fill_value='extrapolate')
    resampled = f(x_new)
    
    # Normalização Z-score
    std = np.std(resampled)
    return (resampled - np.mean(resampled)) / std if std > 1e-8 else resampled - np.mean(resampled)

def carregar_dataset_phyphox(pasta_raiz):
    X_list, y_list, sample_dict = [], [], {}
    
    for caminho in sorted(glob.glob(f"{pasta_raiz}/*.csv")):
        nome = os.path.basename(caminho).lower()
        if "infinito" in nome: label = "infinito"
        elif "shake" in nome: label = "shake"
        elif "tap" in nome: label = "tap"
        else: continue
        
        # Leitura dos dados originais do Phyphox
        df = pd.read_csv(caminho)
        cols = [c for c in df.columns if any(e in c.lower() for e in ['x', 'y', 'z'])]
        x, y, z = df[cols[0]].values, df[cols[1]].values, df[cols[2]].values
        
        # Cálculo da magnitude da aceleração tridimensional
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        processed = preprocess_series(magnitude, TARGET_LENGTH)
        
        X_list.append(processed)
        y_list.append(label)
        
        if label not in sample_dict:
            sample_dict[label] = (magnitude, processed)
            
    return np.array(X_list), np.array(y_list), sample_dict

# Carregar arquivos CSV salvos
X_train, y_train, sample_dict = carregar_dataset_phyphox("meu_dataset_phyphox/treino")
X_test, y_test, _ = carregar_dataset_phyphox("meu_dataset_phyphox/teste")

# ---------------------------------------------------------
# 2. ALGORITMO DTW (DYNAMIC TIME WARPING) & RANDOM FOREST
# ---------------------------------------------------------
def dtw_distance(s1, s2):
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    cost_matrix = cdist(s1.reshape(-1, 1), s2.reshape(-1, 1), metric='sqeuclidean')
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = cost_matrix[i-1, j-1]
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j], 
                dtw_matrix[i, j-1], 
                dtw_matrix[i-1, j-1]
            )
    return np.sqrt(dtw_matrix[n, m])

def predict_1nn_dtw(X_tr, y_tr, X_te):
    preds = []
    for test_sample in X_te:
        distances = [dtw_distance(test_sample, train_sample) for train_sample in X_tr]
        preds.append(y_tr[np.argmin(distances)])
    return np.array(preds)

# Treinamento e Avaliação
y_pred_dtw = predict_1nn_dtw(X_train, y_train, X_test)
acc_dtw = accuracy_score(y_test, y_pred_dtw)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

print("==========================================")
print(f"Acurácia 1NN-DTW: {acc_dtw*100:.2f}%")
print(f"Acurácia Random Forest: {acc_rf*100:.2f}%")
print("==========================================\n")
print("Relatório de Classificação (1NN-DTW):")
print(classification_report(y_test, y_pred_dtw))

# ---------------------------------------------------------
# 3. VISUALIZAÇÃO DOS RESULTADOS
# ---------------------------------------------------------
classes = ["infinito", "shake", "tap"]
# Perfil dos Sinais (Bruto vs Processado)
fig, axes = plt.subplots(3, 2, figsize=(11, 7), sharex='col')
for i, cl in enumerate(classes):
    raw, proc = sample_dict[cl]
    axes[i, 0].plot(raw, color='tab:blue')
    axes[i, 0].set_title(f"Sinal Bruto Phyphox - {cl.capitalize()}")
    axes[i, 0].set_ylabel("m/s²")
    axes[i, 0].grid(True, linestyle='--', alpha=0.5)
    
    axes[i, 1].plot(proc, color='tab:orange')
    axes[i, 1].set_title(f"Sinal Normalizado & Reamostrado - {cl.capitalize()}")
    axes[i, 1].set_ylabel("Z-score")
    axes[i, 1].grid(True, linestyle='--', alpha=0.5)

axes[2, 0].set_xlabel("Amostras Brutas")
axes[2, 1].set_xlabel("Pontos Padronizados (200 pts)")
plt.tight_layout()
plt.savefig("grafico_sinais_phyphox.png", dpi=300)
plt.show()

# Matrizes de Confusão
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
cm_dtw = confusion_matrix(y_test, y_pred_dtw, labels=classes)
sns.heatmap(cm_dtw, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=axes[0])
axes[0].set_title(f"1NN-DTW (Acurácia: {acc_dtw*100:.1f}%)")
axes[0].set_xlabel("Predito")
axes[0].set_ylabel("Real")

cm_rf = confusion_matrix(y_test, y_pred_rf, labels=classes)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', xticklabels=classes, yticklabels=classes, ax=axes[1])
axes[1].set_title(f"Random Forest (Acurácia: {acc_rf*100:.1f}%)")
axes[1].set_xlabel("Predito")
axes[1].set_ylabel("Real")

plt.tight_layout()
plt.savefig("matrizes_confusao.png", dpi=300)
plt.show()