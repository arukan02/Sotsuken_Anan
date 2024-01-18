import json
import os

def change_file_names(json_data, old_prefix, new_prefix):
    for image in json_data["images"]:
        old_file_name = image["file_name"]
        new_file_name = old_file_name.replace(old_prefix, new_prefix)
        image["file_name"] = new_file_name

        # Rename the actual image file
        old_path = os.path.join("images", old_file_name)
        new_path = os.path.join("images", new_file_name)
        os.rename(old_path, new_path)

if __name__ == "__main__":
    json_file_path = "annotations/instances_default.json"  # Replace with the actual path to your JSON file
    old_prefix = "frame_0"
    new_prefix = "frame_1"

    with open(json_file_path, "r") as file:
        data = json.load(file)

    change_file_names(data, old_prefix, new_prefix)

    with open(json_file_path, "w") as file:
        json.dump(data, file, indent=4)

    print(f"File names in {json_file_path} and actual image files updated successfully.")
