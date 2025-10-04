#SCRIPTS
from Model import model

if __name__ == "__main__":
    model.train(
        data = "Dataset\dataset.yaml", #YAML DOSYASININ YOLU
        epochs = 200, 
        imgsz = 640, #HER RESMİ 640X640 PİKSEL OLARAK İŞLER
        batch = 2, 
        name = "Results\MyCustomYoloModel", #EĞİTİM ÇIKTILARININ KAYDEDİLECEĞİ KLASÖR
        workers = 0, #SADECE ANA İŞLEMCİYİ KULLANMASINI SAĞLAR RAM YÜKÜNÜ AZALTIR
        device = 0, # GPU = 0 "CPU" = CPU
        lr0 = 0.005, #LEARNING RATE
        augment = True #VERİYE TRANSFORM UYGULUYOR DEFAULT TRUE
    )