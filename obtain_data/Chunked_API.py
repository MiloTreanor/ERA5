import os
import cdsapi


tmp_pressure_dir = "tmp_pressure"
tmp_single_dir = "tmp_single"

os.makedirs(tmp_pressure_dir, exist_ok=True)
os.makedirs(tmp_single_dir, exist_ok=True)

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
    "pressure_level": ["850", "500"],  # Relevant pressure levels
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

base_request_single = {
                "product_type": ["reanalysis"],
                "variable": ["total_precipitation"],
                "year": ["2022"],
                "month": ["05"],
                "day": [
                    "01", "02", "03",
                    "04", "05", "06",
                    "07", "08", "09",
                    "10", "11", "12",
                    "13", "14", "15",
                    "16", "17", "18",
                    "19", "20", "21",
                    "22", "23", "24",
                    "25", "26", "27",
                    "28", "29", "30",
                    "31"
                ],
                "time": [
                    "00:00", "01:00", "02:00",
                    "03:00", "04:00", "05:00",
                    "06:00", "07:00", "08:00",
                    "09:00", "10:00", "11:00",
                    "12:00", "13:00", "14:00",
                    "15:00", "16:00", "17:00",
                    "18:00", "19:00", "20:00",
                    "21:00", "22:00", "23:00"
                ],
                "data_format": "grib",
                "download_format": "unarchived",
                "area": [89, -32, 17, 40]
            }

# Define the time periods
#years = ["2018", "2019", "2020", "2021"]
#months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

years = ["2022"]
months = ["08", "11", "02"]

# ---------------------------
# Download and combine single-level data in chunks
# ---------------------------

single_files = []

for year in years:
    for month in months:
        req = base_request_single.copy()
        req["year"] = [year]
        req["month"] = [month]
        tmp_filename = os.path.join(tmp_single_dir, f"era5_single_{year}_{month}.grib")
        print(f"Downloading single-level data for {year}-{month}...")
        client.retrieve("reanalysis-era5-single-levels", req).download(tmp_filename)
        single_files.append(tmp_filename)

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


print("Download complete.")

