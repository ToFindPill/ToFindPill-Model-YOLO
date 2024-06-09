# 4.py 에서, csv 파일 이용하여 json 파일 내용 추가한 버전

import torch
import cv2
from pathlib import Path
import sys
import json
import time
import pandas as pd
import shutil
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

# CSV 파일에서 데이터를 불러와 DataFrame으로 변환하는 함수.
def load_csv_data(csv_path):
    return pd.read_csv(csv_path)

# JSON 파일로 저장하기 위한 결과를 수집하는 함수.
def collect_results(image_id, image_path, classification_results, csv_data):
    results = {
        'id': str(image_id),
        'path': str(image_path),
        'candidates': []
    }

    for i, (class_name, confidence) in enumerate(classification_results):
        if confidence == 0:
            continue  # 신뢰도가 0인 결과는 건너뜁니다.

        # ' copy'를 제거
        cleaned_class_name = class_name.replace(' copy', '')

        # CSV 데이터에서 class_name에 해당하는 정보를 찾습니다.
        row = csv_data[csv_data['drug_N'] == cleaned_class_name]

        if not row.empty:
            # 필요한 정보를 CSV에서 추출하여 JSON에 추가합니다.
            drug_info = {
                'id': f"{image_id}-{i+1}",
                'class': cleaned_class_name,
                'rank': i + 1,
                'percentage': f"{confidence * 100:.0f}%",
                'name': row['dl_name'].values[0],
                'image': row['img_key'].values[0],
                'material': row['dl_material'].values[0],
                'description': row['chart'].values[0],
                'type': row['di_class_no'].values[0],
                'code': row['di_etc_otc_code'].values[0],
                'company': row['dl_company'].values[0]
            }
        else:
            # 해당하는 정보가 없으면 기본 정보를 입력합니다.
            drug_info = {
                'id': f"{image_id}-{i+1}",
                'class': cleaned_class_name,
                'rank': i + 1,
                'percentage': f"{confidence * 100:.0f}%"
            }

        results['candidates'].append(drug_info)

    return results

# 분류된 결과를 출력하는 함수.
def classify_image(image_path, model):
    predictions = model(image_path, imgsz=256)
    valid_classes = list(range(252))  # Assuming classes are from 0 to 251

    classification_results = [
        (result.names[class_id], confidence)
        for result in predictions
        for class_id, confidence in zip(result.probs.top5, result.probs.top5conf.tolist())
        if class_id in valid_classes and confidence > 0
    ]

    return classification_results

# output_latest 폴더를 비우는 함수.
def clear_output_latest_folder(output_latest_folder):
    if output_latest_folder.exists():
        for file in output_latest_folder.iterdir():
            if file.is_file():
                file.unlink()  # 파일 삭제
            elif file.is_dir():
                shutil.rmtree(file)  # 디렉터리 삭제

# output 폴더의 파일을 output_latest 폴더로 복사하는 함수.
def copy_to_output_latest(output_folder, output_latest_folder):
    if not output_latest_folder.exists():
        output_latest_folder.mkdir(parents=True, exist_ok=True)
    for item in output_folder.iterdir():
        shutil.copy(item, output_latest_folder / item.name)

# 메인 함수. 모델을 로드하고 이미지에서 객체를 검출합니다.
def main(image_path, detection_model_path, classification_model_path, csv_path):
    # 파인튜닝한 YOLOv5 모델을 로드합니다.
    model = torch.hub.load('/data_2/ace_jungmin/yolov5', 'custom', path=detection_model_path, source='local')
    # 분류 모델 로드
    classify_model = YOLO(classification_model_path)
    print('모델 로드 완료')

    # 탐지 설정 조정
    model.conf = 0.05  # confidence threshold
    model.iou = 0.5  # IoU threshold

    # 이미지를 읽어옵니다.
    img = cv2.imread(image_path)
    img_name = Path(image_path).stem

    # CSV 데이터 로드
    csv_data = load_csv_data(csv_path)

    # 모델을 사용하여 이미지에서 객체를 검출합니다.
    results = model(img)

    # 출력 폴더 생성
    output_folder = create_output_folder('/data_2/ace_pill/database/output')

    # 원본 이미지를 저장합니다.
    original_image_path = output_folder / f'original_pill_image.png'
    save_image(img, original_image_path)

    # 바운딩 박스를 그리기 위한 이미지 초기화
    img_with_boxes = img.copy()

    json_data = {
        'original_image': str(original_image_path),
        'bounding_box_image': None,
        'results': []
    }

    # 검출 결과를 처리하고 잘린 이미지를 저장합니다.
    for i, bbox in enumerate(results.xyxy[0]):  # batch size가 1인 경우를 가정
        x1, y1, x2, y2, conf, cls = map(int, bbox[:6])

        # 이미지에 바운딩 박스를 그립니다.
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 검출된 객체를 이미지에서 잘라냅니다.
        cropped_img = img[y1:y2, x1:x2]

        # 잘린 이미지를 저장합니다.
        crop_img_path = output_folder / f'pill_image_{i}.png'
        save_image(cropped_img, crop_img_path)

        # 저장된 이미지를 분류합니다.
        classification_results = classify_image(crop_img_path, classify_model)

        # JSON 파일을 위한 결과 수집
        json_result = collect_results(i + 1, crop_img_path, classification_results, csv_data)
        json_data['results'].append(json_result)

    # 바운딩 박스가 그려진 이미지를 저장합니다.
    bounding_box_image_path = output_folder / f'bounding_box_pill_image.png'
    save_image(img_with_boxes, bounding_box_image_path)
    json_data['bounding_box_image'] = str(bounding_box_image_path)

    # JSON 파일 저장
    json_output_path = output_folder / 'results.json'
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    print(f"JSON 파일이 생성되었습니다: {json_output_path}")

    # output_latest 폴더 비우기
    output_latest_folder = Path('/data_2/ace_pill/database/output_latest')
    clear_output_latest_folder(output_latest_folder)

    # output 폴더의 파일을 output_latest로 복사
    copy_to_output_latest(output_folder, output_latest_folder)

    print(f"파일이 {output_latest_folder}로 복사되었습니다.")

if __name__ == "__main__":
    start_time = time.time()
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        detection_model_path = '/data_2/ace_jungmin/yolov5/runs/train/yolov5_finetune/weights/last.pt'
        classification_model_path = '/data_2/ace_minjae/yolo_project/runs/classify/real_fine200/weights/best.pt'
        csv_path = '/data_2/ace_pill/pill_dataset_single.csv'
        main(image_path, detection_model_path, classification_model_path, csv_path)
    else:
        print("이미지 경로를 첫 번째 인수로 제공하십시오.")
    
    end_time = time.time()
    print(f"!!!!!!!!!! 소요시간 : {end_time - start_time: .5f} 초 !!!!!!!!!!")
