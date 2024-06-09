import torch
import cv2
from pathlib import Path
import sys
from ultralytics import YOLO

# 출력 폴더를 생성하는 함수. 동일한 이름의 폴더가 이미 존재하면 숫자를 하나씩 증가시켜 새 폴더를 생성합니다.
def create_output_folder(base_folder):
    output_folder = Path(base_folder)
    counter = 0
    while output_folder.exists():
        counter += 1
        output_folder = Path(f"{base_folder}_{counter}")
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder

# 이미지를 지정된 경로에 저장하는 함수.
def save_image(image, path):
    cv2.imwrite(str(path), image)

# 분류된 결과를 출력하는 함수.
def classify_image(image_path, model_path):
    # 분류 모델 로드
    model = YOLO(model_path)
    print('분류 모델 로드 완료')

    # 이미지를 분류합니다.
    predictions = model(image_path, imgsz=256)
    valid_classes = list(range(252))  # Assuming classes are from 0 to 251

    for result in predictions:
        # 가장 높은 확률의 클래스와 확률을 출력
        top1_class_id = result.probs.top1
        top1_confidence = result.probs.top1conf.item()
        print(f'\n이미지 경로: {result.path}')
        
        if top1_class_id in valid_classes:
            print(f"예측된 Top-1 클래스: {result.names[top1_class_id]}, 신뢰도: {top1_confidence:.4f}")
        else:
            print(f"오류: 예측된 클래스 {top1_class_id}가 유효한 범위를 벗어났습니다.")

        # 상위 5개 예측 결과를 출력
        top5_classes = result.probs.top5
        top5_confidences = result.probs.top5conf.tolist()

        print("Top-5 예측 결과:")
        for i, class_id in enumerate(top5_classes):
            if class_id in valid_classes:
                print(f"클래스: {result.names[class_id]}, 신뢰도: {top5_confidences[i]:.4f}")
            else:
                print(f"오류: 예측된 클래스 {class_id}가 유효한 범위를 벗어났습니다.")

# 메인 함수. 모델을 로드하고 이미지에서 객체를 검출합니다.
def main(image_path, detection_model_path, classification_model_path):
    # 파인튜닝한 YOLOv5 모델을 로드합니다.
    model = torch.hub.load('/data_2/ace_jungmin/yolov5', 'custom', path=detection_model_path, source='local')

    # 탐지 설정 조정
    model.conf = 0.05  # confidence threshold
    model.iou = 0.5  # IoU threshold

    # 이미지를 읽어옵니다.
    img = cv2.imread(image_path)
    img_name = Path(image_path).stem

    # 모델을 사용하여 이미지에서 객체를 검출합니다.
    results = model(img)

    # 출력 폴더 생성
    output_folder = create_output_folder('/data_2/ace_pill/database/output')

    # 원본 이미지를 저장합니다.
    original_image_path = output_folder / f'original_pill_image.png'
    save_image(img, original_image_path)

    # 바운딩 박스를 그리기 위한 이미지 초기화
    img_with_boxes = img.copy()

    # 검출 결과를 처리하고 잘린 이미지를 저장합니다.
    for i, bbox in enumerate(results.xyxy[0]):  # batch size가 1인 경우를 가정
        x1, y1, x2, y2, conf, cls = bbox[:6]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # 이미지에 바운딩 박스를 그립니다.
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 검출된 객체를 이미지에서 잘라냅니다.
        cropped_img = img[y1:y2, x1:x2]

        # 잘린 이미지를 저장합니다.
        crop_img_path = output_folder / f'pill_image_{i}.png'
        save_image(cropped_img, crop_img_path)

        # 저장된 이미지를 분류합니다.
        classify_image(crop_img_path, classification_model_path)

    # 바운딩 박스가 그려진 이미지를 저장합니다.
    bounding_box_image_path = output_folder / f'bounding_box_pill_image.png'
    save_image(img_with_boxes, bounding_box_image_path)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        detection_model_path = '/data_2/ace_jungmin/yolov5/runs/train/yolov5_finetune/weights/last.pt'
        classification_model_path = '/data_2/ace_minjae/yolo_project/runs/classify/real_fine200/weights/best.pt'
        main(image_path, detection_model_path, classification_model_path)
    else:
        print("이미지 경로를 첫 번째 인수로 제공하십시오.")
    
    print("!!!!!! 완료 !!!!!!")
