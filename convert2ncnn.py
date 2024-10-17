from ultralytics import YOLO
import time

def convert_model(model):
	
	model_path = "D:/Ultra_ask/v5_ncnn"
	model_name = f"yolov{model}_trained.pt"
	model = YOLO(f"{model_path}/{model_name}")

	start_time = time.time() #Begin time
	# Export the model
	model.export(format="ncnn")

	end_time = time.time()  # Record the end time
	execution_time = end_time - start_time
	print(f"Time Elapsed {execution_time} seconds")
	return execution_time

def convert2engine_log(model_name, execution_time):
	with open (f"./convert2ncnn_log/{model_name}.txt","w") as f:
		f.write(f'Model: {model_name} \n')
		f.write(f'Time:  {execution_time} seconds \n')

model_list = ['5nu','5s6u','5su']
for model in model_list:
	convert2engine_log(model, convert_model(model))