#------------------SADECE BİR GÖRSEL ÜZERİNDE TAHMİN YAPMAK İSTERSEK------------------

#LIBRARIES
from ultralytics import YOLO

if __name__ == '__main__':
    
    best_model_path = "Results\\MyCustomYoloModel2\\weights\\best.pt"  #EN İYİ AĞRILIKLARIN PATHI
    model = YOLO(best_model_path) 
    
    results = model.predict(
        source="RESİMYOLU", # Test etmek istediğiniz yeni görselin TAM YOLU
        conf=0.7, # Güven eşiği (bu değerin üzerindeki tahminleri göster)
        iou=0.05, #TAHMİN KUTULARINDA BİRBİRİNE %10 VEYA ÜZERİ ÖRTÜŞENLERİ SİLER
        save=True  # Sonucu kaydedilen görsel olarak kaydet
    )

    print("Tahminler kaydedildi.")
