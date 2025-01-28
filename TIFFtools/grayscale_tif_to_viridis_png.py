import tkinter as tk
from tkinter import filedialog
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tifffile

# Function to open file dialog and select a grayscale TIFF file
def select_grayscale_tif():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select a Grayscale TIFF File", filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")])
    return file_path

# Function to save file dialog for the output PNG
def select_output_location():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    output_path = filedialog.asksaveasfilename(title="Save as", defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
    return output_path

# Main function to convert grayscale TIFF to viridis colormap PNG
def convert_to_viridis_colormap():
    # Step 1: Select the input file
    input_file = select_grayscale_tif()
    if not input_file:
        print("No file selected.")
        return

    # Step 2: Load the grayscale TIFF using tifffile
    grayscale_array = tifffile.imread(input_file)

    # Step 3: Normalize and apply viridis colormap
    colormap_image = plt.cm.viridis(grayscale_array / np.max(grayscale_array))  # Normalize to [0,1] and apply colormap

    # Step 4: Convert to 8-bit RGBA image
    colormap_image = (colormap_image * 255).astype(np.uint8)  # Scale to [0,255] range
    rgba_image = Image.fromarray(colormap_image, "RGBA")

    # Step 5: Select the output location and save as PNG
    output_file = select_output_location()
    if output_file:
        rgba_image.save(output_file)
        print(f"Image saved to {output_file}")
    else:
        print("No output file selected.")

# Run the function
if __name__ == "__main__":
    convert_to_viridis_colormap()
