#SCRIPTS
from Model import model

if __name__ == "__main__":
    model.train(
        data = "Dataset\dataset.yaml", #YAML DOSYASININ YOLU
        epochs = 10, 
        imgsz = 640, #HER RESMİ 640X640 PİKSEL OLARAK İŞLER
        batch = 4, 
        name = "Results\MyCustomYoloModels2", #EĞİTİM ÇIKTILARININ KAYDEDİLECEĞİ KLASÖR
        workers = 0, #SADECE ANA İŞLEMCİYİ KULLANMASINI SAĞLAR RAM YÜKÜNÜ AZALTIR
        device = 0, # GPU = 0 "CPU" = CPU
        lr0 = 0.0005, #YENİ LEARNING RATE
        half= True,
        augment = True, #VERİYE TRANSFORM UYGULUYOR DEFAULT TRUE
        resume=True
    )