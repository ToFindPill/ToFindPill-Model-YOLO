from ultralytics import YOLO
import sys
sys.path.append('/data_2/ace_minjae/yolo_project/')
# Load a model
model = YOLO("/data_2/ace_minjae/yolo_project/yolo-V8/ultralytics/models/v8/cls/yolov8l-cls.yaml")  # build a new model from YAML
# model = YOLO("./yolov8l-cls.pt")  # load a pretrained model (recommended for training)
# model = YOLO("/data_2/ace_minjae/yolo_project/yolo-V8/ultralytics/models/v8/cls/yolov8l-cls.yaml").load("./yolov8l-cls.pt")  # build from YAML and transfer weights

print('모델 로드 완료')
# Train the model
results = model.train(data="datasets/images", epochs=100, imgsz=256, device=[0, 1], name='./output')

print('모델 학습 완료')

# Validate the model
metrics = model.val()
print('모델 검증 완료')

print(metrics.top1)  # top1 accuracy
print(metrics.top5)  # top5 accuracy

# predictions = model.predict(source="datasets/images/val/K-000573/K-000573_0_0_0_0_60_100_200.png")
# # 예측 결과 출력
# print(predictions)

print('얏호~')