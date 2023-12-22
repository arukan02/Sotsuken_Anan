import os
import json
import shutil
from sklearn.model_selection import train_test_split

# Define the source folder containing all the images
source_folder = 'images'

# Define the destination folders for train and validation images
train_folder = 'train2017'
validation_folder = 'val2017'

# Load the COCO format JSON file
coco_json_file = 'annotations.json'

# Define the ratio for validation data (e.g., 0.2 for 20% validation)
validation_ratio = 0.2

with open(coco_json_file, 'r') as f:
    coco_data = json.load(f)

# Extract image information from the COCO JSON
images = coco_data['images']

# Extract image file names and their corresponding IDs
image_id_to_filename = {image['id']: image['file_name'] for image in images}

# Split the images into train and validation sets
image_ids = list(image_id_to_filename.keys())
train_ids, validation_ids = train_test_split(image_ids, test_size=validation_ratio, random_state=42)

# Create the train and validation folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(validation_folder, exist_ok=True)

# Move images to the appropriate folders and update segmentation data
for image_id in image_ids:
    source_path = os.path.join(source_folder, image_id_to_filename[image_id])
    if image_id in train_ids:
        destination_path = os.path.join(train_folder, image_id_to_filename[image_id])
    else:
        destination_path = os.path.join(validation_folder, image_id_to_filename[image_id])

    # Move the image to the appropriate folder
    try:
        shutil.move(source_path, destination_path)
        print(f"Moved {image_id_to_filename[image_id]} to {'train' if image_id in train_ids else 'validation'} folder.")
    except FileNotFoundError:
        print(f"Image not found: {image_id_to_filename[image_id]}")

# Create separate JSON files for training and validation data
train_data = {
    'info': coco_data['info'],
    'licenses': coco_data['licenses'],
    'images': [image for image in images if image['id'] in train_ids],
    'annotations': [annotation for annotation in coco_data['annotations'] if annotation['image_id'] in train_ids],
    'categories': coco_data['categories']
}

validation_data = {
    'info': coco_data['info'],
    'licenses': coco_data['licenses'],
    'images': [image for image in images if image['id'] in validation_ids],
    'annotations': [annotation for annotation in coco_data['annotations'] if annotation['image_id'] in validation_ids],
    'categories': coco_data['categories']
}

train_json_file = 'custom_train.json'
validation_json_file = 'custom_val.json'

with open(train_json_file, 'w') as train_json:
    json.dump(train_data, train_json)

with open(validation_json_file, 'w') as validation_json:
    json.dump(validation_data, validation_json)

print("Data and JSON files separation complete.")
