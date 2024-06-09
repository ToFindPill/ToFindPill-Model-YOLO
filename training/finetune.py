from ultralytics import YOLO
import sys
sys.path.append('/data_2/ace_minjae/yolo_project/')
# Load a model
# model = YOLO("runs/classify/epoch200/weights/best.pt") 

model = YOLO("./yolov8l-cls.pt") 
print('모델 로드 완료')

# Train the model
results = model.train(data="datasets2/images", epochs=200, imgsz=256, device=[0, 1], name='./output')

print('모델 학습 완료')

# # Validate the model
# metrics = model.val()
# print('모델 검증 완료')

# print(metrics.top1)  # top1 accuracy
# print(metrics.top5)  # top5 accuracy

# predictions = model.predict(source="datasets/images/val/K-000573/K-000573_0_0_0_0_60_100_200.png")
# # 예측 결과 출력
# print(predictions)

print('얏호~')