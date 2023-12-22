import json
import pandas as pd

# Load OpenImages metadata, classes, and annotations
image_metadata = pd.read_csv('default-images-with-rotation.csv')
class_labels = pd.read_csv('class-descriptions.csv')
annotations = pd.read_csv('default-annotations-bbox.csv')

# Create a dictionary to store the COCO dataset structure
coco_data = {
    "licenses": [
        {
            "name": "",
            "id": 0,
            "url": ""
        }
    ],
    "info": {
        "contributor": "",
        "date_created": "",
        "description": "",
        "url": "",
        "version": "",
        "year": ""
    },
    "categories": [
        {
            "id": 0,
            "name": "tomato_flower",
            "supercategory": "N/A"
        }
    ],
    "images": [],
    "annotations": [],
}

# Map OpenImages class IDs to COCO category IDs
class_id_mapping = {}  # Map OpenImages class ID to COCO category ID

# Create a dictionary to count annotations per image
annotations_count = {}  # Key: image_id, Value: count

# Iterate through OpenImages data and convert to COCO format
for index, row in image_metadata.iterrows():
    image_id = row['ImageID']
    file_name = row['ImageName']
    width = row['Width']
    height = row['Height']

    # Create COCO image entry
    coco_image = {
        "id": image_id,
        "file_name": file_name + '.PNG',
        "width": width,
        "height": height,
        "coco_url": "",
        "flickr_url": "",
    }
    coco_data["images"].append(coco_image)

    
    # Process annotations for this image and convert to COCO format
    image_annotations = annotations[annotations['ImageID'] == file_name]
    for _, annotation in image_annotations.iterrows():        
        # Extract relevant information and convert to COCO format
        x, y, w, h = int(annotation['XMin'] * width), int(annotation['YMin'] * height), int(annotation['XMax'] * width), int(annotation['YMax'] * height)
        category_id = class_id_mapping.get(annotation['LabelName'], 0)  # Replace 0 with COCO category ID

        # Create segmentation in the form of polygons
        # You can modify this part to create actual polygon data
        segmentation = [[x, y, x, h, w, h, w, y]]

         # Count annotations per image
        annotations_count[image_id] = annotations_count.get(image_id, 0) + 1

        #add attributes
        attributes = {
            "occluded": False,
            "rotation": 0.0,
            "track_id": annotations_count[image_id]
        }

        # Create COCO annotation entry
        coco_annotation = {
            "id": annotation['ID'],
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [x, y, w - x, h - y],
            "area": (w - x) * (h - y),
            "iscrowd": 0,
            "segmentation": segmentation,
            "attributes": attributes,
        }
        coco_data["annotations"].append(coco_annotation)

# Write the COCO data to a JSON file
with open('annotations.json', 'w') as json_file:
    json.dump(coco_data, json_file)
