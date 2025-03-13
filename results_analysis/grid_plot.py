import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Example: Folder containing images
image_folder = ''  # Replace with your path

# List all image files (Assuming images are in formats like .png, .jpg)
image_files = [
    'images/ADASYN_MS_abalone_9_vs_18_comparison_fold_0.png', 'images/ADASYN_MS_abalone_9_vs_18_comparison_fold_0.png', 'images/ADASYN_MS_abalone_9_vs_18_comparison_fold_0.png',
    'images/ADASYN_MS_abalone_9_vs_18_comparison_fold_1.png', 'images/ADASYN_MS_abalone_9_vs_18_comparison_fold_0.png', 'images/ADASYN_MS_abalone_9_vs_18_comparison_fold_0.png',
    'images/ADASYN_MS_abalone_9_vs_18_comparison_fold_2.png', 'images/ADASYN_MS_abalone_9_vs_18_comparison_fold_0.png', 'images/ADASYN_MS_abalone_9_vs_18_comparison_fold_0.png',
    'images/ADASYN_MS_abalone_9_vs_18_comparison_fold_3.png', 'images/ADASYN_MS_abalone_9_vs_18_comparison_fold_0.png', 'images/ADASYN_MS_abalone_9_vs_18_comparison_fold_0.png',
    'images/ADASYN_MS_abalone_9_vs_18_comparison_fold_4.png', 'images/ADASYN_MS_abalone_9_vs_18_comparison_fold_0.png', 'images/ADASYN_MS_abalone_9_vs_18_comparison_fold_0.png'
]

# Create a 5x3 grid for plotting (5 columns, 3 rows)
fig, axes = plt.subplots(5, 3, figsize=(20, 20))  # 3 rows, 5 columns

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Loop through images and plot
for idx, img_file in enumerate(image_files):
    img_path = os.path.join(image_folder, img_file)
    img = mpimg.imread(img_path)  # Load the image

    axes[idx].imshow(img)
    axes[idx].axis('off')  # Hide axis
    #axes[idx].set_title(f"Image {idx + 1}")

# Hide any unused subplots if less than 15 images
for j in range(len(image_files), 15):
    axes[j].axis('off')

# Adjust layout and display the grid
plt.tight_layout()
plt.subplots_adjust(wspace=-0.05, hspace=-0.0)
plt.savefig("grid.png")
plt.show()
