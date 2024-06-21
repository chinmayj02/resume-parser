import matplotlib.pyplot as plt
import pandas as pd

# Sample data (replace with your actual data)
data = {
    'k': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    'accuracy': [53, 53, 53, 53, 53, 53, 53, 53, 62, 65, 81, 91],
    'precision': [57, 57, 57, 57, 57, 57, 57, 57, 66, 62, 86, 100]
}

df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(df['k'], df['accuracy'], marker='o', linestyle='-', color='b', label='Accuracy')
plt.plot(df['k'], df['precision'], marker='o', linestyle='-', color='r', label='Precision')

plt.title('Metrics vs. Number of Recommendations (k)')
plt.xlabel('Number of Recommendations (k)')
plt.ylabel('Metric Score')
plt.xticks(df['k'])
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot as an image in the current directory
plt.savefig('metrics_vs_number_of_recommendations.png')

plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Example confusion matrix (replace with your actual confusion matrix)
conf_matrix = np.array([[50, 10],
                        [5, 35]])

# Define classes and colors for better visualization
classes = ['Negative', 'Positive']
colors = ['lightblue', 'lightcoral']

plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

# Add text annotations
thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             ha="center", va="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()

# Save the plot as an image in the current directory
plt.savefig('confusion_matrix.png')

plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Data for the bar plot
k_values = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
accuracy_scores = np.array([53, 53, 53, 53, 53, 53, 53, 53, 62, 65, 81, 91])
precision_scores = np.array([57, 57, 57, 57, 57, 57, 57, 57, 66, 62, 86, 100])
list_of_jobs = np.array([8, 8, 8, 8, 8, 8, 8, 8, 7, 6, 4, 3])

# Calculate the width of the bars
bar_width = 0.2

# Set the positions of the bars on the x-axis
r1 = np.arange(len(k_values))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.bar(r1, accuracy_scores, color='b', width=bar_width, edgecolor='grey', label='Accuracy')
plt.bar(r2, precision_scores, color='g', width=bar_width, edgecolor='grey', label='Precision')
plt.bar(r3, list_of_jobs, color='r', width=bar_width, edgecolor='grey', label='List of Jobs')

# Add labels, title, and x-axis tick labels
plt.xlabel('K Values', fontweight='bold')
plt.xticks([r + bar_width for r in range(len(k_values))], k_values)
plt.ylabel('Scores', fontweight='bold')
plt.title('Overall Scores for Different K Values')
plt.legend()

# Show the plot
plt.tight_layout()
plt.savefig('Overall Scores for Different K Values.png')
plt.show()
