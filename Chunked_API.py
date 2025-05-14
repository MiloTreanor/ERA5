import os
import cdsapi


tmp_pressure_dir = "tmp_pressure"

os.makedirs(tmp_pressure_dir, exist_ok=True)

client = cdsapi.Client()


base_request_pressure = {
    "product_type": "reanalysis",
    "variable": [
        "temperature",          # Temperature at multiple levels
        "u_component_of_wind",  # U-wind at multiple levels
        "v_component_of_wind",  # V-wind at multiple levels
        "geopotential",         # Geopotential Height
        "relative_humidity",
        "vertical_velocity"     # Vertical Velocity
    ],
    "pressure_level": ["850", "500", "700"],  # Relevant pressure levels
    "day": [
        "01", "02", "03", "04", "05", "06", "07", "08", "09",
        "10", "11", "12", "13", "14", "15", "16", "17", "18",
        "19", "20", "21", "22", "23", "24", "25", "26", "27",
        "28", "29", "30", "31"
    ],
    "time": [
        "00:00", "01:00", "02:00", "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00", "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00", "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"
    ],
    "data_format": "grib",
    "download_format": "unarchived",
    "area": [89, -32, 17, 40]  # Modify this for your region
}

# Define the time periods
years = ["2018", "2019", "2020", "2021"]
months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

# ---------------------------
# Download and combine single-level data in chunks
# ---------------------------
single_files = []

# Optional: Remove temporary files for single-level data
# for fname in single_files:
#     os.remove(fname)

# ---------------------------
# Download and combine pressure-level data in chunks
# ---------------------------
pressure_files = []

for year in years:
    for month in months:
        req = base_request_pressure.copy()
        req["year"] = [year]
        req["month"] = [month]
        tmp_filename = os.path.join(tmp_pressure_dir, f"era5_pressure_{year}_{month}.grib")
        print(f"Downloading pressure-level data for {year}-{month}...")
        client.retrieve("reanalysis-era5-pressure-levels", req).download(tmp_filename)
        pressure_files.append(tmp_filename)

# Combine all monthly pressure-level files into one final file
final_pressure = "era5_pressure_levels_2018-2021.grib"
with open(final_pressure, "wb") as outfile:
    for fname in sorted(pressure_files):
        with open(fname, "rb") as infile:
            outfile.write(infile.read())

# Optional: Remove temporary files for pressure-level data
# for fname in pressure_files:
#     os.remove(fname)

print("Download and concatenation complete.")