import os
import shutil
import random

# Define paths
source_folder = 'crop_dataset'
train_images_folder = 'datasets/images/train'
val_images_folder = 'datasets/images/val'
train_labels_folder = 'datasets/labels/train'
val_labels_folder = 'datasets/labels/val'

# Class mapping based on provided list
class_mapping = {
    "K-000573": 0, "K-011233": 1, "K-011234": 2, "K-011235": 3, "K-011251": 4,
    "K-011254": 5, "K-011262": 6, "K-011299": 7, "K-011307": 8, "K-011392": 9,
    "K-011396": 10, "K-011399": 11, "K-011416": 12, "K-011425": 13, "K-011426": 14,
    "K-011441": 15, "K-011445": 16, "K-011487": 17, "K-011506": 18, "K-011508": 19,
    "K-011529": 20, "K-011530": 21, "K-011545": 22, "K-011558": 23, "K-011564": 24,
    "K-011567": 25, "K-011569": 26, "K-011570": 27, "K-011579": 28, "K-011590": 29,
    "K-011599": 30, "K-011609": 31, "K-011610": 32, "K-011630": 33, "K-011635": 34,
    "K-011651": 35, "K-011676": 36, "K-011680": 37, "K-011701": 38, "K-011718": 39,
    "K-011763": 40, "K-011764": 41, "K-011822": 42, "K-011825": 43, "K-011835": 44,
    "K-011836": 45, "K-011844": 46, "K-011871": 47, "K-011875": 48, "K-011918": 49,
    "K-011952": 50, "K-024489": 51, "K-024512": 52, "K-024516": 53, "K-024522": 54,
    "K-024527": 55, "K-024586": 56, "K-024587": 57, "K-024603": 58, "K-024608": 59,
    "K-024609": 60, "K-024626": 61, "K-024639": 62, "K-024650": 63, "K-024655": 64,
    "K-024704": 65, "K-024722": 66, "K-024723": 67, "K-024731": 68, "K-024747": 69,
    "K-024770": 70, "K-024783": 71, "K-024784": 72, "K-024806": 73, "K-024811": 74,
    "K-024813": 75, "K-024831": 76, "K-024850": 77, "K-024853": 78, "K-024863": 79,
    "K-024883": 80, "K-024904": 81, "K-024914": 82, "K-024915": 83, "K-024924": 84,
    "K-024928": 85, "K-024939": 86, "K-024941": 87, "K-024949": 88, "K-024950": 89,
    "K-025003": 90, "K-025005": 91, "K-025030": 92, "K-025040": 93, "K-025060": 94,
    "K-025075": 95, "K-025077": 96, "K-025078": 97, "K-025090": 98, "K-025097": 99,
    "K-025103": 100, "K-025458": 101, "K-030157": 102, "K-030158": 103, "K-030173": 104,
    "K-030188": 105, "K-030198": 106, "K-030199": 107, "K-030225": 108, "K-030244": 109,
    "K-030283": 110, "K-030285": 111, "K-030286": 112, "K-030295": 113, "K-030301": 114,
    "K-030310": 115, "K-030326": 116, "K-030347": 117, "K-030356": 118, "K-030365": 119,
    "K-030372": 120, "K-030378": 121, "K-030384": 122, "K-030409": 123, "K-030411": 124,
    "K-030422": 125, "K-030459": 126, "K-030473": 127, "K-030474": 128, "K-030478": 129,
    "K-030488": 130, "K-030505": 131, "K-030507": 132, "K-030511": 133, "K-030512": 134,
    "K-030520": 135, "K-030524": 136, "K-030528": 137, "K-030535": 138, "K-030542": 139,
    "K-030552": 140, "K-030555": 141, "K-030562": 142, "K-030577": 143, "K-030590": 144,
    "K-030598": 145, "K-030600": 146, "K-030615": 147, "K-030616": 148, "K-030617": 149,
    "K-030636": 150, "K-030649": 151, "K-033352": 152, "K-033373": 153, "K-033376": 154,
    "K-033398": 155, "K-033421": 156, "K-033422": 157, "K-033448": 158, "K-033479": 159,
    "K-033497": 160, "K-033511": 161, "K-033513": 162, "K-033543": 163, "K-033556": 164,
    "K-033557": 165, "K-033595": 166, "K-033607": 167, "K-033615": 168, "K-033621": 169,
    "K-033622": 170, "K-033623": 171, "K-033628": 172, "K-033636": 173, "K-033682": 174,
    "K-033688": 175, "K-033711": 176, "K-033720": 177, "K-033736": 178, "K-033747": 179,
    "K-033805": 180, "K-033820": 181, "K-033823": 182, "K-033856": 183, "K-033874": 184,
    "K-033877": 185, "K-033878": 186, "K-033886": 187, "K-033895": 188, "K-033913": 189,
    "K-033927": 190, "K-033930": 191, "K-033960": 192, "K-033963": 193, "K-033973": 194,
    "K-033979": 195, "K-033989": 196, "K-033992": 197, "K-034011": 198, "K-034053": 199,
    "K-034055": 200, "K-034058": 201, "K-037576": 202, "K-037585": 203, "K-037608": 204,
    "K-037644": 205, "K-037653": 206, "K-037654": 207, "K-037656": 208, "K-037660": 209,
    "K-037667": 210, "K-037671": 211, "K-037733": 212, "K-037747": 213, "K-037748": 214,
    "K-037755": 215, "K-037757": 216, "K-037761": 217, "K-037775": 218, "K-037777": 219,
    "K-037789": 220, "K-037809": 221, "K-037822": 222, "K-037827": 223, "K-037854": 224,
    "K-037864": 225, "K-037872": 226, "K-037875": 227, "K-037908": 228, "K-037909": 229,
    "K-037910": 230, "K-037931": 231, "K-037941": 232, "K-037959": 233, "K-037963": 234,
    "K-037975": 235, "K-037991": 236, "K-037999": 237, "K-038018": 238, "K-038019": 239,
    "K-038025": 240, "K-038057": 241, "K-038076": 242, "K-038104": 243, "K-038107": 244,
    "K-038131": 245, "K-038132": 246, "K-038134": 247, "K-038140": 248, "K-038144": 249,
    "K-038149": 250, "K-038162": 251
}

# Create destination folders if they don't exist
os.makedirs(train_images_folder, exist_ok=True)
os.makedirs(val_images_folder, exist_ok=True)
os.makedirs(train_labels_folder, exist_ok=True)
os.makedirs(val_labels_folder, exist_ok=True)

# Get list of all class folders
class_folders = [folder for folder in os.listdir(source_folder) if folder.startswith('K-')]

# Loop through each class folder
for class_folder in class_folders:
    class_folder_path = os.path.join(source_folder, class_folder)
    images = [file for file in os.listdir(class_folder_path) if file.endswith('.png')]

    # Shuffle the images
    random.shuffle(images)

    # Split images into 80% train and 20% val
    split_index = int(0.8 * len(images))
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Create class subdirectories in train and val folders
    os.makedirs(os.path.join(train_images_folder, class_folder), exist_ok=True)
    os.makedirs(os.path.join(val_images_folder, class_folder), exist_ok=True)
    os.makedirs(os.path.join(train_labels_folder, class_folder), exist_ok=True)
    os.makedirs(os.path.join(val_labels_folder, class_folder), exist_ok=True)

    # Copy images to train folder and create label files
    for image in train_images:
        src_path = os.path.join(class_folder_path, image)
        dest_path = os.path.join(train_images_folder, class_folder, image)
        shutil.copy2(src_path, dest_path)
        
        # Create label file for the image
        label_content = str(class_mapping[class_folder])
        label_file = image.replace('.png', '.txt')
        label_dest_path = os.path.join(train_labels_folder, class_folder, label_file)
        with open(label_dest_path, 'w') as f:
            f.write(label_content)

    # Copy images to val folder and create label files
    for image in val_images:
        src_path = os.path.join(class_folder_path, image)
        dest_path = os.path.join(val_images_folder, class_folder, image)
        shutil.copy2(src_path, dest_path)

        # Create label file for the image
        label_content = str(class_mapping[class_folder])
        label_file = image.replace('.png', '.txt')
        label_dest_path = os.path.join(val_labels_folder, class_folder, label_file)
        with open(label_dest_path, 'w') as f:
            f.write(label_content)

print("Dataset split into training and validation sets successfully, with label files created.")
