from openai import OpenAI
import pandas as pd
import random
import sys

client = OpenAI(
    base_url="http://localhost:1234/v1", 
    api_key="1234" 
)

topics = ["Tarih", "Coğrafya", "Spor", "Sinema ve Sanat", "Bilim ve Teknoloji", "Edebiyat"]
dataset_filename = "genel_kultur_veri_seti.csv"
data_rows = []

for i in range(50):
    konu = random.choice(topics)
    sys.stdout.flush()
    
    prompt = f"""
    Sen genel kültür yarışması sunucususun. Konu: {konu}.
    Lütfen sadece ve sadece şu formatta cevap ver:
    
    Soru: [Soru buraya]
    Doğru Cevap: [Doğru cevap buraya]
    Yanlış Cevap: [Yanlış cevap buraya]
    """
    
    try:
        completion = client.chat.completions.create(
            model="model-identifier", # TEK COSMOS VAR
            messages=[
                {"role": "system", "content": "Sen yardımcı bir asistansın."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        
        raw_output = completion.choices[0].message.content
        
        data_rows.append({"Konu": konu, "Ham_Cikti": raw_output})
        
    except Exception as e:
        print(f"\nHata: {e}")
        break

# Kaydetme
if data_rows:
    df = pd.DataFrame(data_rows)
    df.to_csv(dataset_filename, sep=";", index=False, encoding="utf-8-sig")
    print(f"\nBitti")