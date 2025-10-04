#LIBRARIES
from ultralytics import YOLO

# model = YOLO("yolov8l.pt") #FINETUNE YAPMADAN ÖNCE KULLANDIĞIM MODEL EN BAŞTA BUNUNLA EĞİTTİM

model = YOLO("Results\\MyCustomYoloModels2\\weights\\best.pt") #FINE TUNE YAPTIĞIM MODEL