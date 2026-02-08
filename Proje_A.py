import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# --- AYARLAR ---
TRAIN_FILE = "egitim_kucuk.pkl" # Eğitim verisi
TEST_FILE = "test.pkl"          # Test verisi
EPOCHS = 100                    # Dönem sayısı
LEARNING_RATE = 0.01            # Öğrenme katsayısı

# --- 1. VERİ YÜKLEME ---
def load_data(filename):
    if not os.path.exists(filename):
        print(f"HATA: '{filename}' bulunamadı.")
        exit()
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    # Soru ve Cevap vektörlerini birleştir (Concat)
    X = np.hstack([data["soru_vec"], data["cevap_vec"]])
    # Sonuna 1 ekleme (w0 için)
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    y = data["labels"].reshape(-1, 1)
    return X, y

X_train, y_train = load_data(TRAIN_FILE)
X_test, y_test = load_data(TEST_FILE)
print(f"   Eğitim Seti: {X_train.shape}, Test Seti: {X_test.shape}")

# --- 2. MATEMATİKSEL FONKSİYONLAR ---
def tanh(x): 
    # Aktivasyon fonksiyonu
    return np.tanh(x)

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def accuracy(y_true, y_pred):
    # İşaret kontrolü: Pozitifse +1, Negatifse -1
    return np.mean(np.sign(y_pred) == y_true)

# --- 3. EĞİTİM MOTORU ---
def train(opt_name, w_init):
    w = w_init.copy()
    hist = {"loss": [], "acc": [], "time": []}
    start = time.time()
    
    # Optimizer Bellekleri
    m, v, cache = np.zeros_like(w), np.zeros_like(w), np.zeros_like(w)
    beta1, beta2, eps = 0.9, 0.999, 1e-8 # Adam/RMSProp Parametreleri
    t = 0
    
    for epoch in range(EPOCHS):
        # Forward Pass
        z = np.dot(X_train, w)
        y_pred = tanh(z)
        
        # Gradient Hesaplama (Türev)
        # Hata * Tanh Türevi (1 - y^2) * Giriş
        err = y_pred - y_train
        dz = err * (1 - y_pred**2) 
        grad = np.dot(X_train.T, dz) / len(X_train)
        
        # --- GÜNCELLEME ALGORİTMALARI ---
        
        if opt_name == "GD":
            w -= LEARNING_RATE * grad
            
        elif opt_name == "SGD":
            idx = np.random.randint(0, len(X_train))
            x_s, y_s = X_train[idx:idx+1], y_train[idx:idx+1]
            z_s = np.dot(x_s, w)
            y_p_s = tanh(z_s)
            g_s = np.dot(x_s.T, (y_p_s - y_s) * (1 - y_p_s**2))
            w -= LEARNING_RATE * g_s
            
        elif opt_name == "Adam":
            t += 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad**2)
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            w -= LEARNING_RATE * m_hat / (np.sqrt(v_hat) + eps)
        #BONUS  

        elif opt_name == "Adagrad": 
            cache += grad**2
            w -= LEARNING_RATE * grad / (np.sqrt(cache) + eps)
            
        elif opt_name == "RMSProp": 
            v = 0.9 * v + 0.1 * (grad**2)
            w -= LEARNING_RATE * grad / (np.sqrt(v) + eps)

        # Test Seti ile Başarı Ölçümü       
        test_pred = tanh(np.dot(X_test, w))
        hist["loss"].append(mse_loss(y_test, test_pred))
        hist["acc"].append(accuracy(y_test, test_pred))
        hist["time"].append(time.time() - start)
        
    return hist

# --- 4. DENEYLERİ ÇALIŞTIR ---
optimizers = ["GD", "SGD", "Adam", "Adagrad", "RMSProp"]
results = {opt: {"loss": [], "acc": [], "time": []} for opt in optimizers}

for i in range(5):
    print(f"   Tur {i+1}/5...")
    # 5 farklı w değeri için rastgele başlangıç
    w_init = np.random.randn(X_train.shape[1], 1) * 0.01 
    
    for opt in optimizers:
        h = train(opt, w_init)
        results[opt]["loss"].append(h["loss"])
        results[opt]["acc"].append(h["acc"])
        results[opt]["time"].append(h["time"])

# Ortalamaları Al
avg = {opt: {k: np.mean(v, axis=0) for k, v in res.items()} for opt, res in results.items()}

# --- 5. GRAFİKLER  ---
print("3. Grafikler 'Odev_A_Final.png' dosyasına çiziliyor...")
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Renkler
colors = {"GD":"blue", "SGD":"orange", "Adam":"green", "Adagrad":"purple", "RMSProp":"red"}

# 1. Epoch vs Loss
for opt in optimizers: axs[0, 0].plot(avg[opt]["loss"], label=opt, color=colors[opt])
axs[0, 0].set_title("Epoch vs Loss")
axs[0, 0].set_ylabel("MSE Loss"); axs[0, 0].legend()

# 2. Epoch vs Accuracy (Başarı)
for opt in optimizers: axs[0, 1].plot(avg[opt]["acc"], label=opt, color=colors[opt])
axs[0, 1].set_title("Epoch vs Accuracy")
axs[0, 1].set_ylabel("Accuracy"); axs[0, 1].legend()

# 3. Süre vs Loss
for opt in optimizers: axs[1, 0].plot(avg[opt]["time"], avg[opt]["loss"], label=opt, color=colors[opt])
axs[1, 0].set_title("Süre (sn) vs Loss")
axs[1, 0].set_xlabel("Time (s)"); axs[1, 0].set_ylabel("Loss"); axs[1, 0].legend()

# 4. Süre vs Accuracy
for opt in optimizers: axs[1, 1].plot(avg[opt]["time"], avg[opt]["acc"], label=opt, color=colors[opt])
axs[1, 1].set_title("Süre (sn) vs Accuracy")
axs[1, 1].set_xlabel("Time (s)"); axs[1, 1].set_ylabel("Accuracy"); axs[1, 1].legend()

plt.tight_layout()
plt.savefig("KUCUK.png")
plt.show()
