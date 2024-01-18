from PIL import Image
import os

def cut_and_resize_image(input_path, output_path, cut_size, final_size):
    #opeb the original image
    original_image = Image.open(input_path)

    #cut the image with (0,0) coordinates
    cut_image = original_image.crop((0,0, cut_size[0], cut_size[1]))

    #resize the cut image
    final_image = cut_image.resize(final_size)

    #save the final image
    final_image.save(output_path)

def process_images_in_folder(input_folder, output_folder, cut_size, final_size):
    #ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    #Iterate through each file in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".PNG"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            cut_and_resize_image(input_path, output_path, cut_size, final_size)

if __name__ == "__main__":
    input_folder_path = "images"
    output_folder_path = "cut_images"
    cut_size = (1350, 800)
    final_size = (1920, 1080)

    process_images_in_folder(input_folder_path, output_folder_path, cut_size, final_size)
    print(f"all images in folder {input_folder_path} has been cut and resized")
