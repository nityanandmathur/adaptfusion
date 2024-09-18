import cupy as cp
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
import numpy as np

# Function to process an image and extract R, G, B channels using CuPy
def process_image(image_path):
    image = Image.open(image_path)
    image = cp.array(image)  # Convert image to a CuPy array
    R = image[:, :, 0].flatten()
    G = image[:, :, 1].flatten()
    B = image[:, :, 2].flatten()
    # print('R:', R)
    return R, G, B

# Function to plot PDF (note: matplotlib does not directly use CuPy)
def plot_pdf(channel_data, color, label):
    plt.hist(cp.asnumpy(channel_data), bins=256, density=True, color=color, alpha=0.6, label=label)

# Function to save individual histograms
def save_histogram(channel_data, color, label, filename):
    # Convert CuPy array to NumPy array
    channel_data_np = cp.asnumpy(channel_data)

    # Check for invalid values
    if np.any(np.isnan(channel_data_np)) or np.any(np.isinf(channel_data_np)):
        print(f"Warning: Invalid values encountered in {label} channel data. Skipping histogram.")
        return

    # Check if the sum of the data is zero
    if np.sum(channel_data_np) == 0:
        print(f"Warning: Sum of {label} channel data is zero. Skipping histogram.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(cp.asnumpy(channel_data), bins=256, density=True, color=color, alpha=0.6)
    plt.title(f'PDF of {label} Channel')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Probability Density')
    plt.savefig(filename)
    plt.close()

# Directory containing images
image_dir = '/home/btech/nityanand.mathur/adaptfusion/results-nu-city/001'

# Initialize lists to hold channel data for all images using CuPy arrays
all_R = cp.array([])
all_G = cp.array([])
all_B = cp.array([])

# Process each image in the directory
for filename in tqdm(os.listdir(image_dir)):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
        image_path = os.path.join(image_dir, filename)
        R, G, B = process_image(image_path)
        all_R = cp.concatenate((all_R, R))
        all_G = cp.concatenate((all_G, G))
        all_B = cp.concatenate((all_B, B))

# Plot combined PDFs for all images
plt.figure(figsize=(10, 6))
plot_pdf(all_R, 'red', 'Red Channel')
plot_pdf(all_G, 'green', 'Green Channel')
plot_pdf(all_B, 'blue', 'Blue Channel')

save_folder = '/home/btech/nityanand.mathur/adaptfusion/plots/nu_city/001'

plt.title('PDFs of R, G, B Channels for All Images')
plt.xlabel('Pixel Intensity')
plt.ylabel('Probability Density')
plt.legend()
plt.savefig(f'{save_folder}/pdf_plot_all_images.png')
plt.close()

# Save individual histograms for all images
save_histogram(all_R, 'red', 'Red', f'{save_folder}/red_channel_histogram_all_images.png')
save_histogram(all_G, 'green', 'Green', f'{save_folder}/green_channel_histogram_all_images.png')
save_histogram(all_B, 'blue', 'Blue', f'{save_folder}/blue_channel_histogram_all_images.png')
