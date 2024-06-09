from ultralytics import YOLO
import sys
sys.path.append('/data_2/ace_minjae/yolo_project/')

# Load a model
model = YOLO("runs/classify/epoch100/weights/best.pt")  # load a pretrained model (recommended for training)
print('모델 로드 완료')

predictions = model(source="test_images/gray/drug_*", imgsz=256)

valid_classes = list(range(252))  # Assuming classes are from 0 to 251

# Extract and print classification results
for result in predictions:
    # Get the top predicted class and probability
    top1_class_id = result.probs.top1
    top1_confidence = result.probs.top1conf.item()
    print('\npath:', result.path)

    if top1_class_id in valid_classes:
        print(f"Predicted Top-1 Class: {result.names[top1_class_id]}, Confidence: {top1_confidence:.4f}")
    else:
        print(f"Error: Predicted class {top1_class_id} is out of valid range.")

    # Get top 5 predictions
    top5_classes = result.probs.top5
    top5_confidences = result.probs.top5conf.tolist()

    print("Top-5 Predictions:")
    for i, class_id in enumerate(top5_classes):
        if class_id in valid_classes:
            print(f"Class: {result.names[class_id]}, Confidence: {top5_confidences[i]:.4f}")
        else:
            print(f"Error: Predicted class {class_id} is out of valid range.")