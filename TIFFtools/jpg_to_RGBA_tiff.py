from PIL import Image
import numpy as np


def convert_jpg_to_tiff_with_alpha(input_jpg, output_tiff):
    # Open the JPG image
    img = Image.open(input_jpg).convert("RGBA")  # Ensure RGBA mode

    # Extract RGB channels and create an alpha channel (fully opaque)
    r, g, b, _ = img.split()
    alpha = Image.new("L", img.size, 255)  # Fully opaque

    # Merge the channels back into an RGBA image
    img_rgba = Image.merge("RGBA", (r, g, b, alpha))

    # Save as TIFF
    img_rgba.save(output_tiff, format="TIFF")
    print(f'Saved {output_tiff}')


# Example usage
input_jpg = "/Users/thowe/MinersAI Dropbox/Science/Geo Data/Chile/Project COBALTERA (Volt)_Chilean Cobalt/Private/Unprocessed/Drilling/02 _Drillhole Collar Map_Rosa Amelia Mine Area.jpg"
output_tiff = "/Users/thowe/MinersAI Dropbox/Science/Geo Data/Chile/Project COBALTERA (Volt)_Chilean Cobalt/Private/Unprocessed/Drilling/02 _Drillhole Collar Map_Rosa Amelia Mine Area.tif"
convert_jpg_to_tiff_with_alpha(input_jpg, output_tiff)
