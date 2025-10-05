# Custom-Object-Detection-with-YOLO
# [TR]
## Projenin Amacı
Bu projeyi, YOLOv8 modelinin önceden eğitilmiş sürümünü kullanarak, modelin orijinalinde yer almayan özel sınıfları (Mouse, Custom Cup, Adapter, Toy Car) transfer learning kullanarak hem gerçek zamanlı hem de fotoğraflar üzerinden tanıyabilmesini sağlamak amacıyla geliştirdim. Proje sürecinde kullanılan tüm verileri derleyerek bu dört nesne için kapsamlı bir veri seti oluşturdum ve bu setin etiketleme işlemlerini eksiksiz tamamladım.

## 📸 Veri Seti Oluşturma Süreci
- Toplamda 2100’den fazla görüntü topladım ve bu görüntülerle 4 ayrı nesne için (Mouse, Custom Cup, Adapter, Toy Car) veri seti oluşturdum.
- Modelin genelleme yeteneğini arttırmak ve nesneleri daha iyi algılayabilmesi için tüm nesneleri, 3 farklı arka plan, 3 farklı aydınlatma koşulu ve çeşitli açılarla çektim.
- Her bir görüntüyü manuel olarak etiketledim (bounding box / labeling) ayrıca tüm verileri train / validation klasörlerine dikkatlice ayırdım.
- Bu sayede modelin, farklı ortam ve açılarda da doğru tahmin yapabilmesi hedefledim.

## 💻 Kullanılan Teknolojiler
- Python 3.11.8
- PyTorch
- Torchvision
- OpenCV
- YOLOv8s (Ultralytics)

## ⚙️ Kurulum
GEREKLİ KÜTÜPHANELERİ KURUN
```bash
pip install ultralytics
```
```bash
pip install opencv-python
```

## 🚀 Çalıştırma
```
└── /dataset
    ├── /train
    │   ├── /images      # Eğitim görselleri buraya
    │   └── /labels      # Eğitim etiketleri (.txt) buraya
    │
    └── /val
        ├── /images      # Doğrulama (validation) görselleri buraya
        └── /labels      # Doğrulama etiketleri (.txt) buraya
    dataset.yaml         # Veri seti yapılandırma dosyası
```

1. Dataset'inizi yerleştirirken bu dosya düzenine uymaya özen gösterin.
2. **Model.py** dosyasına kullanmak istediğiniz YOLO modelini yazabilirsiniz.
3. Datasetinize göre **dataset.yaml** dosyasını güncelleyin.
4. **ModelTrain.py** dosyasını çalıştırıp modelinizi eğitin. (Bu dosyayıda sistem gereksinimlerinize ve datasetinize göre güncelleyin)
5. Model eğitildikten sonra **Predict.py** dosyasına resim vererek tahmin yapmasını sağlayın.
6. **RealtimePredict.py** dosyası ile gerçek zamanlı olarak modelin performansını ölçebilirsiniz. (Bu dosyayıda gerekliliklerinize ve datasetinize göre güncelleyin)

## 📸 Ekran Görüntüleri 
| Adaptör | Oyuncak Araba | 
| :---------------------------------: | :------------------------: |
| ![2](https://github.com/user-attachments/assets/9aa887cd-ef4b-4cfd-b9e3-e9b9ebb07e6b)| ![3](https://github.com/user-attachments/assets/558e923a-dfbe-4dd2-8ea8-a6391bc7214b)
| Kupa | Mouse | 
| ![4](https://github.com/user-attachments/assets/f2303103-2152-48b8-b590-7ffc5f070f1d)| ![5](https://github.com/user-attachments/assets/6e15bb5d-84bf-4dec-b583-36ed391d148e)

<div align="center">
  <h3>Tüm Nesneler</h3>
  <img src="https://github.com/user-attachments/assets/a11f5017-caf0-47fb-b098-2b17dfe63f60" alt="TÜM NESNELER" width="80%" />
</div>

## 📺 Uygulama Videosu
▶️ [Watch Project Video on YouTube](https://www.youtube.com/watch?v=CnQmaLw0khs)

## BU PROJE HİÇBİR ŞEKİLDE TİCARİ AMAÇ İÇERMEMEKTEDİR.

# [EN]
## Project Objective
I developed this project using the pre-trained version of the YOLOv8 model to enable it to recognize custom classes (Mouse, Custom Cup, Adapter, Toy Car) which were not included in the original model through transfer learning, both in real-time and from images. During the project, I compiled all the data used to create a comprehensive dataset for these four objects and completed the labeling process thoroughly.

## 📸 Dataset Creation Process
- I collected over 2,100 images in total and used them to create a dataset for four distinct objects (Mouse, Custom Cup, Adapter, Toy Car).
- To enhance the model's generalization ability and object detection performance, I captured all objects under three different backgrounds, three varying lighting conditions, and from multiple angles.
- I manually labeled (bounding box/labeling) every image and meticulously organized all the data into training and validation folders.
- My goal was to ensure that the model would be capable of making accurate predictions across diverse environments and viewing angles.

 ## 💻 Technologies Used
- Python 3.11.8
- PyTorch
- Torchvision
- OpenCV
- YOLOv8s (Ultralytics)

## ⚙️ Installation
INSTALL THE REQUIRED LIBRARIES
```bash
pip install ultralytics
```
```bash
pip install opencv-python
```

## 🚀 How to Run
```
└── /dataset
    ├── /train
    │   ├── /images     # Training images go here
    │   └── /labels     # Training annotations (.txt) go here
    │
    └── /val
        ├── /images    # Validation images go here  
        └── /labels    # Validation annotations (.txt) go here
    dataset.yaml       # Dataset configuration file
```

- Take care to adhere to this file structure when placing your dataset.
- You can specify the YOLO model you wish to use in the Model.py file.
- Update the dataset.yaml file according to your dataset.
- Run the ModelTrain.py file to train your model. (Also update this file according to your system requirements and dataset.)
- After the model is trained, provide an image to the Predict.py file to get predictions.
- You can measure the model's performance in real-time using the RealtimePredict.py file. (Also update this file according to your requirements and dataset.)


## 📸 Screenshots
| Adapter | Toy Car | 
| :---------------------------------: | :------------------------: |
| ![2](https://github.com/user-attachments/assets/9aa887cd-ef4b-4cfd-b9e3-e9b9ebb07e6b)| ![3](https://github.com/user-attachments/assets/558e923a-dfbe-4dd2-8ea8-a6391bc7214b)
| Custom Cup | Mouse | 
| ![4](https://github.com/user-attachments/assets/f2303103-2152-48b8-b590-7ffc5f070f1d)| ![5](https://github.com/user-attachments/assets/6e15bb5d-84bf-4dec-b583-36ed391d148e)

<div align="center">
  <h3>All Objects</h3>
  <img src="https://github.com/user-attachments/assets/a11f5017-caf0-47fb-b098-2b17dfe63f60" alt="TÜM NESNELER" width="80%" />
</div>

## 📺 Demo Video
▶️ [Watch Project Video on YouTube](https://www.youtube.com/watch?v=CnQmaLw0khs)

## THIS PROJECT IS STRICTLY FOR NON-COMMERCIAL PURPOSES



