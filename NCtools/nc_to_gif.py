import xarray as xr
import matplotlib.pyplot as plt
import os
from PIL import Image

# Specify file paths
file_path = r"C:/Users/TyHow/MinersAI Dropbox/Tyler Howe/ICB_data/with_faults/with_faults.nc"
gif_path = r"C:/Users/TyHow/MinersAI Dropbox/Tyler Howe/ICB_data/with_faults/output_gif.gif"

# Open the dataset
dataset = xr.open_dataset(file_path)

# Extract the necessary variables
layers = dataset["combined_layers"]
x_coords = dataset["x"]
y_coords = dataset["y"]

# Calculate the aspect ratio
x_range = x_coords.values.max() - x_coords.values.min()
y_range = y_coords.values.max() - y_coords.values.min()
aspect_ratio = x_range / y_range

# Define plot limits for consistent borders
x_min, x_max = x_coords.min().item(), x_coords.max().item()
y_min, y_max = y_coords.min().item(), y_coords.max().item()

# Create a temporary folder for images
frames_folder = os.path.join(os.path.dirname(gif_path), "temp_gif_frames").replace("\\", "/")
os.makedirs(frames_folder, exist_ok=True)

# Verify the folder exists
if not os.path.isdir(frames_folder):
    raise FileNotFoundError(f"Temporary frames folder not found: {frames_folder}")

# Plot each layer and save frames
frames = []
for i in range(layers.shape[0]):
    plt.figure(figsize=(8, 8 / aspect_ratio))
    plt.imshow(
        layers[i, :, :],
        extent=[x_min, x_max, y_min, y_max],  # Ensure consistent extent
        cmap="viridis",
        origin="lower",
    )
    plt.xlim(x_min, x_max)  # Consistent x-axis limits
    plt.ylim(y_min, y_max)  # Consistent y-axis limits
    plt.title(f"Layer {i+1}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    #plt.colorbar(label="Value")
    plt.tight_layout()

    # Save the frame in the frames_folder
    frame_path = os.path.join(frames_folder, f"frame_{i:03d}.png").replace("\\", "/")
    plt.savefig(frame_path, bbox_inches='tight')
    plt.close()
    frames.append(frame_path)

# Create the GIF with Pillow
images = [Image.open(frame_path) for frame_path in frames]
images[0].save(
    gif_path,
    save_all=True,
    append_images=images[1:],
    duration=250,  # 1000ms = 1 second per frame
    loop=0  # Infinite loop
)

# Clean up temporary frames
for frame_path in frames:
    os.remove(frame_path)
os.rmdir(frames_folder)

print(f"GIF saved as {gif_path}")
