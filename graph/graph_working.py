import matplotlib.pyplot as plt
import json

# Read data from the text file
data = []
with open('log.txt', 'r') as file:
    for line in file:
        record = json.loads(line)
        data.append(record)

# Select fields to plot
fields_to_plot = ['train_loss', 'test_loss', 'train_class_error', 'test_class_error']

# Extract data for selected fields
epochs = [record['epoch'] for record in data]
field_data = {field: [record[field] for record in data] for field in fields_to_plot}

# Plot data for each field
for field in fields_to_plot:
    plt.plot(epochs, field_data[field], label=field)

# Add labels and legend
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Metrics')
plt.legend()
plt.grid(True)

# Show plot
plt.show()
