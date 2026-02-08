import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
import os

files = {
    "kucuk_egitim": ("genel_kultur_veri_seti_kucuk.csv", "egitim_kucuk.pkl"),
    "buyuk_egitim": ("genel_kultur_veri_seti_buyuk.csv", "egitim_buyuk.pkl"),
    "test":         ("test_veri_seti.csv", "test.pkl")
}

model_name = "ytu-ce-cosmos/turkish-e5-large"
print("Model yükleniyor")
model = SentenceTransformer(model_name)

def process_file(input_csv, output_pkl):
    if not os.path.exists(input_csv):
        return
    df = pd.read_csv(input_csv, sep=";")
    processed_data = []
    # CSV Ayrıştırma
    for i, row in df.iterrows():
        col = row.keys()[1] 
        lines = str(row[col]).split('\n')
        
        soru = next((l.split("Soru:")[-1].strip() for l in lines if "Soru:" in l), "")
        dogru = next((l.split("Doğru Cevap:")[-1].strip() for l in lines if "Doğru Cevap:" in l), "")
        yanlis = next((l.split("Yanlış Cevap:")[-1].strip() for l in lines if "Yanlış Cevap:" in l), "")
        
        if soru and dogru and yanlis:
            # +1 (İyi Cevap)
            processed_data.append({"Soru": soru, "Cevap": dogru, "Label": 1.0})
            # -1 (Kötü Cevap)
            processed_data.append({"Soru": soru, "Cevap": yanlis, "Label": -1.0})

    # Vektörleştirme
    print(f"   {len(processed_data)} örnek vektörleştiriliyor...")
    soru_vec = model.encode([p["Soru"] for p in processed_data], show_progress_bar=True)
    cevap_vec = model.encode([p["Cevap"] for p in processed_data], show_progress_bar=True)
    labels = np.array([p["Label"] for p in processed_data])

    # Kaydetme
    with open(output_pkl, 'wb') as f:
        pickle.dump({"soru_vec": soru_vec, "cevap_vec": cevap_vec, "labels": labels}, f)

# Tüm dosyaları işle
for key, (csv, pkl) in files.items():
    process_file(csv, pkl)