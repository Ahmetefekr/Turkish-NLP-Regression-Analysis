import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# --- AYARLAR ---
TRAIN_FILE = "egitim_kucuk.pkl" 
EPOCHS = 50                    
LEARNING_RATE = 0.01

# --- 1. VERİYİ YÜKLEME ---
if not os.path.exists(TRAIN_FILE):
    print(f"HATA: '{TRAIN_FILE}' bulunamadı!")
    exit()

with open(TRAIN_FILE, 'rb') as f:
    data = pickle.load(f)

# Veriyi Hazırla
X = np.hstack([data["soru_vec"], data["cevap_vec"]])
X = np.hstack([X, np.ones((X.shape[0], 1))]) # Bias
y = data["labels"].reshape(-1, 1)


# --- 2. FONKSİYONLAR ---
def tanh(x): return np.tanh(x)

# --- 3. KAYIT ALAN EĞİTİM FONKSİYONU ---
def train_and_record(opt_name, w_init):
    w = w_init.copy()
    w_history = [w.flatten()]
    
    # Optimizer Bellekleri
    m, v, cache = np.zeros_like(w), np.zeros_like(w), np.zeros_like(w)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    
    for epoch in range(EPOCHS):
        # Forward
        z = np.dot(X, w)
        y_pred = tanh(z)
        
        # Gradient
        err = y_pred - y
        dz = err * (1 - y_pred**2)
        grad = np.dot(X.T, dz) / len(X)
        
        # Güncelleme
        if opt_name == "GD":
            w -= LEARNING_RATE * grad
        elif opt_name == "SGD":
            idx = np.random.randint(0, len(X))
            x_s, y_s = X[idx:idx+1], y[idx:idx+1]
            z_s = np.dot(x_s, w)
            y_p_s = tanh(z_s)
            g_s = np.dot(x_s.T, (y_p_s - y_s) * (1 - y_p_s**2))
            w -= LEARNING_RATE * g_s
        elif opt_name == "Adam":
            t += 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad**2)
            m_hat, v_hat = m/(1-beta1**t), v/(1-beta2**t)
            w -= LEARNING_RATE * m_hat / (np.sqrt(v_hat) + eps)
        elif opt_name == "Adagrad":
            cache += grad**2
            w -= LEARNING_RATE * grad / (np.sqrt(cache) + eps)
        elif opt_name == "RMSProp":
            v = 0.9 * v + 0.1 * (grad**2)
            w -= LEARNING_RATE * grad / (np.sqrt(v) + eps)
            
        # Her epoch sonunda ağırlığı kaydet
        w_history.append(w.flatten())
        
    return np.array(w_history)

# --- 4. GÖRSELLEŞTİRME DÖNGÜSÜ ---
optimizers = ["GD", "SGD", "Adam", "Adagrad", "RMSProp"]

for opt in optimizers:
    all_trajectories = []
    
    for i in range(5):
        print(f"   Deneme {i+1}/5...")
        w_init = np.random.randn(X.shape[1], 1) * 0.05
        w_path = train_and_record(opt, w_init)
        all_trajectories.append(w_path)
    
    # Verileri T-SNE için birleştir
    combined_data = np.vstack(all_trajectories)
    
    n_samples = combined_data.shape[0]
    perp = min(30, n_samples - 1) 
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp, init='pca', learning_rate='auto')
    w_2d = tsne.fit_transform(combined_data)
    
    # --- ÇİZİM ---
    plt.figure(figsize=(10, 8))
    start_idx = 0
    colors = ['r', 'g', 'b', 'c', 'm']
    
    for i in range(5):
        # Her denemenin uzunluğu (Epoch sayısı + 1)
        length = len(all_trajectories[i])
        end_idx = start_idx + length
        
        # O denemenin 2D koordinatlarını al
        traj_2d = w_2d[start_idx:end_idx]
        
        # Çizgi çiz
        plt.plot(traj_2d[:, 0], traj_2d[:, 1], marker='.', linestyle='-', color=colors[i], label=f'Başlangıç {i+1}', alpha=0.6)
        
        # Başlangıç (Yuvarlak) ve Bitiş (Yıldız) noktalarını işaretle
        plt.scatter(traj_2d[0, 0], traj_2d[0, 1], c=colors[i], s=100, marker='o', edgecolors='k') # Start
        plt.scatter(traj_2d[-1, 0], traj_2d[-1, 1], c=colors[i], s=200, marker='*', edgecolors='k') # End
        
        start_idx = end_idx

    plt.title(f"{opt} Optimizasyon Yörüngeleri (T-SNE 2D)")
    plt.xlabel("T-SNE Boyut 1")
    plt.ylabel("T-SNE Boyut 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = f"Odev_B_{opt}.png"
    plt.savefig(filename)
    plt.close()


