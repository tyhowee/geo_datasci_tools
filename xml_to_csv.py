import xml.etree.ElementTree as ET
import pandas as pd


def convert_usgs_xml_to_csv(xml_path: str, csv_output: str):
    """
    Converts a USGS field station XML file into a CSV file for use in QGIS.

    Parameters:
    xml_path (str): Path to the input XML file.
    csv_output (str): Path to the output CSV file.

    Output:
    Saves a CSV file containing field station ID, longitude, and latitude.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        data = []
        for sample in root.findall("sample"):
            title = sample.find("title").text.strip()
            coordinates = sample.find("coordinates").text.strip()

            # Split coordinates into longitude and latitude
            try:
                lon, lat = map(float, coordinates.split(","))
                data.append({"ID": title, "Longitude": lon, "Latitude": lat})
            except ValueError:
                print(f"Skipping entry with invalid coordinates: {title}")

        # Convert to DataFrame and save as CSV
        df = pd.DataFrame(data)
        df.to_csv(csv_output, index=False)

        print(f"CSV file saved as {csv_output}")

    except Exception as e:
        print(f"Error processing XML file: {e}")


if __name__ == "__main__":
    input_file = '/Users/thowe/MinersAI Dropbox/Tyler Howe/AK_sample_project/UNPROCESSED/misc/alaska_field_stations_57c71fb3e4b0f2f0cebed0f0.xml'
    output_file = "/Users/thowe/MinersAI Dropbox/Tyler Howe/AK_sample_project/UNPROCESSED/misc/alaska_field_stations.csv"
    convert_usgs_xml_to_csv(input_file, output_file)
    print("Done processing XMLs.")
