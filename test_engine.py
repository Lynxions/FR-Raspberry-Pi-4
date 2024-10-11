from ultralytics import YOLO
import cv2
import time
import numpy as np


def current_milli_time():
    return round(time.time() * 1000)

def valuate_video(model):
    model_path = "./"
    model_name = f"yolov{model}_trained.pt"
    model = YOLO(f"{model_path}/{model_name}")

    # results = model.predict(source="./images/img_1.jpg", show=True, stream=True,device="cuda",save=True, save_txt=True,rect=True,imgsz=[960,1600])
    #results = model.predict(source="./videos/test_video.mp4", show=True, stream=True, device="cuda")
    #results = model.predict(source="./videos/test_video.mp4", stream=True, device="cuda")
    results = model.predict(source= "0", show = True, stream=True, device="cuda")



    timestamp_list = []
    wait_time = 20
    conf_list = []
    for result in results:
        conf = result.boxes.conf

        if len(conf) == 1:
            conf_list.append(float(conf))

        timestamp_list.append(current_milli_time())
        if (cv2.waitKey(wait_time) & 0xFF) == ord('q'):
            break



    duration_list = np.diff(timestamp_list)
    fps_list = []
    for duration in duration_list:
        fps_list.append(1/((duration-wait_time)/1000))
    # print(fps_list[1:])

    avg_fps = sum(fps_list[1:]) / (len(fps_list)-1)
    avg_conf = sum(conf_list[1:]) / (len(conf_list)-1)

    print(f'Valuating {model_name}')
    print(f'Average FPS: {avg_fps} FPS')
    print(f'Average Confidence: {avg_conf}')

    # return [model_name,avg_fps,avg_conf]
    log_valuation(model_name, avg_fps, avg_conf)

def log_valuation(model_name, avg_fps, avg_conf):
    with open (f"./valuate_log/{model_name}.txt","w") as f:
        f.write(f'avg_fps:  {avg_fps}\n')
        f.write(f'avg_conf: {avg_conf}\n')


   
#25.04    
model_list = ['8n']
for model in model_list:
    valuate_video(model)
