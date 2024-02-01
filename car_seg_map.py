from PIL import Image
import numpy as np
import torch
from skimage import measure
import matplotlib.pyplot as plt


image_path_mask = "/Users/gohyixian/Desktop/disentangled_carla/masks/blackcar_day_both/rect_001_3_r5000.png"

img_mask = Image.open(image_path_mask)   # RGB [0-255], HWC
semantic_map = np.array(img_mask)


# Define the target pixel value: vehicle class is [0, 0, 142]
target_pixel = np.array([0, 0, 142])

# Create a binary mask where the semantic map equals the target pixel
binary_mask = np.all(semantic_map == target_pixel, axis=-1)


# make the binary mask to RGB shape
# latent mask is 4 in depth
binary_mask = np.stack((binary_mask,)*3, axis=-1)
binary_mask = binary_mask.transpose(2,0,1)
print(binary_mask.shape)


# make it c*h*w
example = dict()
example["binary_image_mask"] = torch.from_numpy(binary_mask)


# check if the values are either 0 or 1.
is_binary = ((example["binary_image_mask"] == 0) | (example["binary_image_mask"] == 1)).all()
assert is_binary.item() ==  True

#                                             C, H,   W
print(example["binary_image_mask"].shape)   # 3, 600, 800
# return example

labeled_image, num_labels = measure.label(binary_mask[0], connectivity=2, return_num=True)

# Find the largest connected component
largest_component_label = np.argmax(np.bincount(labeled_image.flat)[1:]) + 1

# Create a new binary image with only the largest component
largest_component_image = labeled_image == largest_component_label

# Find the coordinates of the bounding box around the largest component
coords = np.column_stack(np.where(largest_component_image))

# Get the top-left and bottom-right coordinates
top_left = np.min(coords, axis=0)
bottom_right = np.max(coords, axis=0)


# Calculate normalized coordinates (as a percentage of image size)
height, width = binary_mask[0].shape   # 600, 800
normalized_top_left = (top_left[0] / height, top_left[1] / width)
normalized_bottom_right = (bottom_right[0] / height, bottom_right[1] / width)

print(normalized_top_left, normalized_bottom_right)

# Display the results using Matplotlib
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.set_title('Original Boolean Tensor')
ax1.imshow(binary_mask[0], cmap='gray')

ax2.set_title('Largest Connected Component')
ax2.imshow(largest_component_image, cmap='gray')

ax3.set_title('Bounding Box')
ax3.imshow(largest_component_image, cmap='gray')
ax3.add_patch(plt.Rectangle((top_left[1], top_left[0]), bottom_right[1] - top_left[1], bottom_right[0] - top_left[0], linewidth=2, edgecolor='r', facecolor='none'))

# Add text annotations with normalized coordinates
ax3.text(top_left[1], top_left[0] - 5, f'({normalized_top_left[1]}, {normalized_top_left[0]})', color='r', fontsize=8, ha='center')
ax3.text(bottom_right[1], bottom_right[0] + 15, f'({normalized_bottom_right[1]}, {normalized_bottom_right[0]})', color='r', fontsize=8, ha='center')

fig.text(0.01, 0.1, f"Top Left: ({top_left[1]}, {top_left[0]})    Bottom Right: ({bottom_right[1]}, {bottom_right[0]})")
fig.text(0.01, 0.04, f"Top Left: ({normalized_top_left[1]}, {normalized_top_left[0]})    Bottom Right: ({normalized_bottom_right[1]}, {normalized_bottom_right[0]})")

plt.show()

# Print the normalized top-left and bottom-right coordinates
print(f"Normalized Top-left coordinates: ({normalized_top_left[1]:.2%}, {normalized_top_left[0]:.2%})")
print(f"Normalized Bottom-right coordinates: ({normalized_bottom_right[1]:.2%}, {normalized_bottom_right[0]:.2%})")