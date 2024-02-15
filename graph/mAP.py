import matplotlib.pyplot as plt
import json
import numpy as np

# Read data from the text file
data = []
with open('log.txt', 'r') as file:
    for line in file:
        record = json.loads(line)
        data.append(record)

# Extract mAP values and epochs for each file
mAP_values_per_file = []
for record in data:
    if 'test_coco_eval_bbox' in record:
        mAP_values_per_file.append(record['test_coco_eval_bbox'])

# Calculate mean mAP value across all epochs for each file
mean_mAP_per_file = [np.mean(mAP_values) for mAP_values in mAP_values_per_file]

# Plot mean mAP across all epochs for each file as a single line graph
plt.plot(mean_mAP_per_file)

# Add labels and title
plt.xlabel('Epoch')
plt.ylabel('Mean Average Precision (mAP)')
plt.title('Mean Average Precision (mAP) Across All Epochs')

# Show plot
plt.grid(True)
plt.show()
