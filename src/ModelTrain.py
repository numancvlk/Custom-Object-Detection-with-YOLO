#SCRIPTS
from Model import model

if __name__ == "__main__":
    model.train(
        data = "Dataset\dataset.yaml", #YAML DOSYASININ YOLU
        epochs = 50, 
        imgsz = 640, #HER RESMİ 640X640 PİKSEL OLARAK İŞLER
        batch = 8, 
        name = "MODELİNKAYDEDİLECEĞİYOL", #EĞİTİM ÇIKTILARININ KAYDEDİLECEĞİ KLASÖR
        workers = 0, #SADECE ANA İŞLEMCİYİ KULLANMASINI SAĞLAR RAM YÜKÜNÜ AZALTIR
        device = 0, # GPU = 0 "CPU" = CPU
        lr0 = 0.001, #LEARNING RATE
        half= True, #GPU BELLEĞİNDEKİ YÜKÜ AZALTIR AMA MODELİN KARARLILIĞINI ETKİLEYEBİLİR
        augment = True, #VERİYE TRANSFORM UYGULUYOR DEFAULT TRUE
        resume=False,
        optimizer="AdamW",
        cos_lr=True,
        patience=15, #15 EPOCH BOYUNCA DEĞİŞİM OLMAZSSA EĞİTİMİ DURDURUR
    )