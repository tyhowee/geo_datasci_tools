from pdf2image import convert_from_path
from PIL import Image


def pdf_to_rgba_tiff(pdf_path: str, output_tiff: str, dpi: int = 300):
    """
    Converts a PDF to an RGBA TIFF while retaining transparency and resolution.

    :param pdf_path: Path to the input PDF.
    :param output_tiff: Path to the output TIFF file.
    :param dpi: DPI resolution for conversion (default: 300).
    """
    images = convert_from_path(pdf_path, dpi=dpi)

    # Convert each page to RGBA and save as TIFF
    images_rgba = [img.convert("RGBA") for img in images]

    # Save as a multi-page TIFF if more than one page exists
    images_rgba[0].save(
        output_tiff, save_all=True, append_images=images_rgba[1:], format="TIFF"
    )


if __name__ == "__main__":
    input_file = '/Users/thowe/MinersAI Dropbox/Tyler Howe/AK_sample_project/UNPROCESSED/geology/Geologic Map of Southeastern Alaska.pdf'
    output_file = '/Users/thowe/MinersAI Dropbox/Tyler Howe/AK_sample_project/UNPROCESSED/geology/Geologic Map of Southeastern Alaska.tif'
    pdf_to_rgba_tiff(pdf_path=input_file, output_tiff=output_file)
    print("Done processing PDFs.")
