import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Directory containing images
# image_dir = "/export/livia/home/vision/Mnoori/old_home/data/segmentation/cityscapes/leftImg8bit/val/"
# image_dir = "/export/livia/home/vision/Mnoori/old_home/data/segmentation/coco_stuff164k/images/val2017/"
image_dir = "/export/livia/home/vision/Mnoori/old_home/data/segmentation/only_val_segmentation/final_data/VOC2012/JPEGImages/"

# Initialize lists to store width, height, and ratio of each image
widths = []
heights = []
ratios = []
orientation = {"horizontal": 0, "vertical": 0}

# Loop over all files in the directory and subdirectories with tqdm progress bar
for root, _, files in os.walk(image_dir):
    for image_name in tqdm(files, desc="Processing Images"):
        image_path = os.path.join(root, image_name)
        
        # Only process if it is a file and has a valid image extension
        if os.path.isfile(image_path) and image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            with Image.open(image_path) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)
                ratios.append(width / height)
                
                # Count orientation based on width and height
                if width > height:
                    orientation["horizontal"] += 1
                elif height > width:
                    orientation["vertical"] += 1

# Convert lists to numpy arrays for easy calculation of statistics
widths = np.array(widths)
heights = np.array(heights)
ratios = np.array(ratios)

# Calculate paired statistics
paired_statistics = {
    "Height": {
        "Min": {"height": np.min(heights), "width": widths[np.argmin(heights)]},
        "Max": {"height": np.max(heights), "width": widths[np.argmax(heights)]},
    },
    "Width": {
        "Min": {"width": np.min(widths), "height": heights[np.argmin(widths)]},
        "Max": {"width": np.max(widths), "height": heights[np.argmax(widths)]},
    }
}

# Calculate individual statistics for width, height, and ratio
individual_statistics = {
    "Width": {
        "Mean": np.mean(widths),
        "Median": np.median(widths),
        "Standard Deviation": np.std(widths)
    },
    "Height": {
        "Mean": np.mean(heights),
        "Median": np.median(heights),
        "Standard Deviation": np.std(heights)
    },
    "Ratio (Width/Height)": {
        "Mean": np.mean(ratios),
        "Median": np.median(ratios),
        "Standard Deviation": np.std(ratios)
    }
}

# Display results
print("Paired Image Size Statistics (Width and Height):")
for dimension, stats in paired_statistics.items():
    print(f"\n{dimension} Statistics:")
    for stat, values in stats.items():
        print(f"  {stat}: {values}")

print("\nIndividual Image Size Statistics:")
df_stats = pd.DataFrame(individual_statistics)
print(df_stats)

# Orientation summary
print("\nOrientation Counts:")
print(f"  Horizontal: {orientation['horizontal']}")
print(f"  Vertical: {orientation['vertical']}")

# Display unique sizes and their counts
unique_sizes, counts = np.unique(list(zip(widths, heights)), axis=0, return_counts=True)
size_distribution = pd.DataFrame({"Size": unique_sizes.tolist(), "Count": counts})
print("\nUnique Size Distribution:")
print(size_distribution.sort_values(by="Count", ascending=False))

# Plotting the distribution of aspect ratios
plt.figure(figsize=(10, 6))
plt.hist(ratios, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel("Aspect Ratio (Width/Height)")
plt.ylabel("Frequency")
plt.title("Distribution of Image Aspect Ratios")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Save plot to file
plt.savefig("aspect_ratio_distribution.png")
print("\nAspect ratio distribution plot saved as 'aspect_ratio_distribution.png'")
