import csv
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib.cm import get_cmap

# Function to calculate the Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Load data from the CSV file
data = []
with open('prep_yoko.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)

    # Skip the header line
    next(csvreader)

    for row in csvreader:
        data.append(row)

# Dictionary to store object IDs and their coordinates
objects = {}


# Image size (adjust these values according to your image dimensions)
image_width = 1920
image_height = 1080

# Threshold for considering two objects as the same
threshold = 45.0 / image_height  # Normalize the threshold

# Set up the figure size to match the image size
fig, ax = plt.subplots()

# Get a colormap and create an iterable of colors
cmap = get_cmap('tab20')  # You can choose any other colormap tab20
colors = cycle(cmap.colors)

# Loop through each frame
for row in data:
    frame_id, *frame_data = row
    frame_id = int(frame_id)

    # Extract object coordinates from string format and normalize
    frame_objects = [list(map(float, coords.strip('()').split(','))) for coords in frame_data]
    normalized_frame_objects = [(coord[0] / image_width, coord[1] / image_height) for coord in frame_objects]

    # Ensure all objects have the same number of coordinates in a frame
    if all(len(coords) == 2 for coords in normalized_frame_objects):
        # Loop through objects in the current frame
        for obj_id, obj_coords in enumerate(normalized_frame_objects):
            obj_id += 1  # Object IDs start from 1
            matched = False

            # Compare the current object with existing objects
            for existing_obj_id, existing_obj_coords in objects.items():
                if euclidean_distance(obj_coords, existing_obj_coords[-1]) < threshold:
                    objects[existing_obj_id].append(obj_coords)
                    matched = True
                    break

            # If not matched, create a new object
            if not matched:
                objects[obj_id] = [obj_coords]

# Show the movement of each object and generate a graph with original size
for obj_id, obj_coords_list in objects.items():
    if obj_coords_list:
        print(f"Object {obj_id} movement:")
        x_coords, y_coords = zip(*obj_coords_list)
        original_x_coords = [coord * image_width for coord in x_coords]
        original_y_coords = [(coord) * image_height for coord in y_coords]
        color = next(colors)
        ax.plot(original_x_coords, original_y_coords, label=f"Object {obj_id}", color=color)


# Set axis limits to match image dimensions
# ax.set_xlim(0, image_width + 200)
# ax.set_ylim(0, image_height +200)

plt.gca().invert_yaxis()  # Invert the Y-axis
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.legend()
plt.show()