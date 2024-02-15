import matplotlib.pyplot as plt
import pandas as pd
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

# Convert mean mAP values to pandas DataFrame for EWMA calculation
df = pd.DataFrame(mean_mAP_per_file)

# Calculate EWMA of mean mAP values
ewm_col = 0  # Adjust the exponential decay factor as needed
coco_eval_ewma = df.ewm(com=ewm_col).mean()

# Plot mean mAP across all epochs for each file with EWMA
plt.plot(coco_eval_ewma)

# Add labels and title
plt.xlabel('Epoch')
plt.ylabel('Exponentially Weighted Mean Average Precision (mAP)')
plt.title('Exponentially Weighted Mean Average Precision (mAP) Across All Epochs')

# Show plot
plt.grid(True)
plt.show()

##THERE'S NO DIFFERENCE WITH THE mAP.py
##THERE'S NO DIFFERENCE WITH THE mAP.py
##THERE'S NO DIFFERENCE WITH THE mAP.py
##THERE'S NO DIFFERENCE WITH THE mAP.py