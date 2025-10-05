#LIBRARIES
import cv2 as cv
import tkinter as tk
from ultralytics import YOLO

def getWantedClasses():
    classNames = []
    if adapterVar.get():
        classNames.append("ADAPTER")
    if mouseVar.get():
        classNames.append("MOUSE")
    if customCupVar.get():
        classNames.append("CUSTOM_CUP")
    if toyCarVar.get():
        classNames.append("TOY_CAR")
    return classNames

video = cv.VideoCapture(0)

root = tk.Tk()
root.title("MY YOLO FILTER")
root.geometry("200x200")

adapterVar = tk.IntVar(value=0)
mouseVar = tk.IntVar(value=0)
customCupVar = tk.IntVar(value=0)
toyCarVar = tk.IntVar(value=0)

tk.Checkbutton(root,text="Adapter",variable=adapterVar).pack(anchor="w")
tk.Checkbutton(root,text="Mouse",variable=mouseVar).pack(anchor="w")
tk.Checkbutton(root, text="Custom Cup", variable=customCupVar).pack(anchor="w")
tk.Checkbutton(root, text="Toy Car", variable=toyCarVar).pack(anchor="w")

best_model_path = "Results\\MyCustomYoloModel2\\weights\\best.pt"  #EN İYİ AĞRILIKLARIN PATHI
model = YOLO(best_model_path) 

def updateFrame():
    isTrue, frame = video.read()

    if not isTrue:
        root.destroy()
        video.release()
        cv.destroyAllWindows()
        return
    frameResized = cv.resize(frame,(640,640))
    results = model(frameResized,
                    imgsz=640,
                    conf= 0.7,
                    iou=0.4) #%40'üzeri oranda örtüşen tahminleri siliyor.

    for result in results:
        mask = [result.names[int(cls)] in getWantedClasses() for cls in result.boxes.cls]
        result.boxes = result.boxes[mask]

    detectionFrame = results[0].plot()
    cv.imshow("YOLO CUSTOM DETECTION",detectionFrame)

    if cv.waitKey(1) != 27:
        root.after(1,updateFrame)
    else:
        root.destroy()
        cv.destroyAllWindows()
        video.release()

root.after(1,updateFrame)
root.mainloop()
