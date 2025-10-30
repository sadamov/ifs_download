#!/usr/bin/env python3
"""
Bulk download script for IFS data across multiple date ranges.
Downloads both ensemble and control IFS data for specified time periods.
"""

import argparse
import ast
import os
from datetime import datetime

import earthkit.data
import xarray as xr
from earthkit.data import settings


def create_fields_file(output_path, model_name="graphcast"):
    """Create a fields.txt file with default parameters if ai-models is not available."""

    # Default parameters for graphcast (commonly used)
    # These can be adjusted based on your specific needs
    default_fields = {
        "grid": [0.25, 0.25],
        "area": [90, -180, -90, 180],  # Global
        "pressure_levels": [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
        "pressure_level_params": ["u", "v", "t", "q", "z"],
        "single_level_params": ["2t", "10u", "10v", "msl", "tp", "z"],
    }

    # Create directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    fields_content = f"""grid: {default_fields["grid"]}
area: {default_fields["area"]}

pressure_levels: {default_fields["pressure_levels"]}
pressure_level_params: {default_fields["pressure_level_params"]}

single_level_params: {default_fields["single_level_params"]}
"""

    fields_path = os.path.join(output_path, "fields.txt")
    with open(fields_path, "w") as f:
        f.write(fields_content)

    print(f"Created fields.txt at {fields_path}")
    return fields_path


def parse_date_range(date_range_str):
    """Parse date range string like '2023-01-02T00:2023-01-08T23' into start and end dates."""
    start_str, end_str = date_range_str.split(":")
    start_date = datetime.strptime(start_str, "%Y-%m-%dT%H")
    end_date = datetime.strptime(end_str, "%Y-%m-%dT%H")
    return start_date, end_date


def download_ifs_ensemble(output_dir, start_date, num_days, interval=6, model_name="graphcast"):
    """Download IFS ensemble data."""

    # Set cache directory
    settings.set("user-cache-directory", "/tmp/earthkit-cache")

    # Read parameters from fields.txt
    date_str = start_date.strftime("%Y%m%d%H%M")
    path = os.path.join(output_dir, date_str, model_name)

    # Create fields.txt if it doesn't exist
    if not os.path.exists(os.path.join(path, "fields.txt")):
        create_fields_file(path, model_name)

    with open(os.path.join(path, "fields.txt"), "r") as f:
        lines = f.readlines()

    grid = ast.literal_eval(lines[0].split(": ")[1].strip())
    area = ast.literal_eval(lines[1].split(": ")[1].strip())
    pressure_levels = ast.literal_eval(lines[3].split(": ")[1].strip())
    pressure_level_params = ast.literal_eval(lines[4].split(": ")[1].strip())
    single_level_params = ast.literal_eval(lines[6].split(": ")[1].strip())

    # Extract date components
    year = start_date.strftime("%Y")
    month = start_date.strftime("%m")
    day = start_date.strftime("%d")
    hour = start_date.strftime("%H")

    # Build the request
    request = {
        "area": area,
        "class": "od",
        "date": f"{year}-{month}-{day}",
        "expver": "1",
        "grid": grid,
        "levtype": "sfc",
        "number": "1/to/50/by/1",
        "param": single_level_params,
        "step": f"0/to/{num_days * 24}/by/{interval}",
        "stream": "enfo",
        "expect": "any",
        "time": hour,
        "type": "pf",
    }

    # Define chunking separately for pressure-level and single-level data.
    # Note: 'time' and 'surface' are not dimensions in these datasets; avoid chunking by them.
    chunks_pl = {
        "number": 1,
        "step": 1,
        "isobaricInhPa": 1,
        "latitude": -1,
        "longitude": -1,
    }
    chunks_surface = {
        "number": 1,
        "step": 1,
        "latitude": -1,
        "longitude": -1,
    }

    print(f"Downloading ensemble data for {date_str}...")

    # Check if output already exists
    output_file = f"{path}/ifs_ens.zarr"
    if os.path.exists(output_file):
        print(f"Output already exists: {output_file}, skipping...")
        return

    # Retrieve the single level data
    ds_single = earthkit.data.from_source("mars", request, lazily=True)
    ds_single = (
        ds_single.to_xarray(chunks=chunks_surface).drop_vars("valid_time").chunk(chunks_surface)
    )

    # Split the "number" dimension into chunks
    number_chunks = [f"{i}/to/{i + 9}/by/1" for i in range(1, 51, 10)]

    # Retrieve the pressure level data in chunks because of MARS size limits
    for i, number_chunk in enumerate(number_chunks):
        request.update({
            "levtype": "pl",
            "levelist": pressure_levels,
            "param": pressure_level_params,
            "number": number_chunk,
        })
        ds_pressure_chunk = earthkit.data.from_source("mars", request, lazily=True)

        shortnames = list(set(ds_pressure_chunk.metadata("shortName")))
        special_vars = ["r"]
        normal_vars = [var for var in shortnames if var not in special_vars]

        # Convert to xarray and chunk
        ds_normal = ds_pressure_chunk.sel(shortName=normal_vars).to_xarray(chunks=chunks_pl)
        ds_special = ds_pressure_chunk.sel(shortName=special_vars).to_xarray(chunks=chunks_pl)
        ds_combined = xr.merge([ds_normal, ds_special]).chunk(chunks_pl).drop_vars("valid_time")

        # Write to Zarr with correct append_dim
        ds_combined.to_zarr(
            output_file,
            consolidated=True,
            mode="w" if i == 0 else "a",
            append_dim="number" if i > 0 else None,
        )
        print(f"Added chunk {i + 1}/{len(number_chunks)}")

    print("Adding Surface data")
    ds_single.drop_vars(["z"] if "z" in ds_single.variables else []).to_zarr(
        output_file, consolidated=True, mode="a"
    )
    print(f"Completed ensemble download for {date_str}")


def download_ifs_control(output_dir, start_date, num_days, interval=6, model_name="graphcast"):
    """Download IFS control data."""

    # Set cache directory
    settings.set("user-cache-directory", "/tmp/earthkit-cache")

    # Read parameters from fields.txt
    date_str = start_date.strftime("%Y%m%d%H%M")
    path = os.path.join(output_dir, date_str, model_name)

    # Create fields.txt if it doesn't exist
    if not os.path.exists(os.path.join(path, "fields.txt")):
        create_fields_file(path, model_name)

    with open(os.path.join(path, "fields.txt"), "r") as f:
        lines = f.readlines()

    grid = ast.literal_eval(lines[0].split(": ")[1].strip())
    area = ast.literal_eval(lines[1].split(": ")[1].strip())
    pressure_levels = ast.literal_eval(lines[3].split(": ")[1].strip())
    pressure_level_params = ast.literal_eval(lines[4].split(": ")[1].strip())
    single_level_params = ast.literal_eval(lines[6].split(": ")[1].strip())

    # Extract date components
    year = start_date.strftime("%Y")
    month = start_date.strftime("%m")
    day = start_date.strftime("%d")
    hour = start_date.strftime("%H")

    # Build the request
    request = {
        "area": area,
        "class": "od",
        "date": f"{year}-{month}-{day}",
        "expver": "1",
        "grid": grid,
        "levtype": "sfc",
        "param": single_level_params,
        "step": f"0/to/{num_days * 24}/by/{interval}",
        "stream": "enfo",
        "expect": "any",
        "time": hour,
        "type": "cf",
    }

    # Define chunking separately for pressure-level and single-level data for control run.
    chunks_pl = {
        "step": 1,
        "isobaricInhPa": 1,
        "latitude": -1,
        "longitude": -1,
    }
    chunks_surface = {
        "step": 1,
        "latitude": -1,
        "longitude": -1,
    }

    print(f"Downloading control data for {date_str}...")

    # Check if output already exists
    output_file = f"{path}/ifs_control.zarr"
    if os.path.exists(output_file):
        print(f"Output already exists: {output_file}, skipping...")
        return

    # Retrieve the single level data
    ds_single = earthkit.data.from_source("mars", request, lazily=True)
    ds_single = (
        ds_single.to_xarray(chunks=chunks_surface).drop_vars("valid_time").chunk(chunks_surface)
    )

    # Retrieve the pressure level data
    request.update({
        "levtype": "pl",
        "levelist": pressure_levels,
        "param": pressure_level_params,
    })
    ds_pressure = earthkit.data.from_source("mars", request, lazily=True)

    shortnames = list(set(ds_pressure.metadata("shortName")))
    special_vars = ["r"]
    normal_vars = [var for var in shortnames if var not in special_vars]

    ds_normal = ds_pressure.sel(shortName=normal_vars).to_xarray(chunks=chunks_pl)
    ds_special = ds_pressure.sel(shortName=special_vars).to_xarray(chunks=chunks_pl)
    ds_combined = xr.merge([ds_normal, ds_special]).chunk(chunks_pl).drop_vars("valid_time")

    print("Writing to zarr")
    ds_combined.to_zarr(
        output_file,
        consolidated=True,
        mode="w",
    )

    print("Adding Surface data")
    ds_single.drop_vars(["z"] if "z" in ds_single.variables else []).to_zarr(
        output_file, consolidated=True, mode="a"
    )
    print(f"Completed control download for {date_str}")


def main():
    parser = argparse.ArgumentParser(description="Bulk download IFS data for multiple date ranges")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/capstor/store/cscs/swissai/a122/IFS",
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="graphcast",
        help="AI model name (graphcast, fourcastnetv2-small, etc.)",
    )
    parser.add_argument(
        "--interval", type=int, default=6, help="Time step in hours between each analysis time"
    )
    parser.add_argument(
        "--download-type",
        type=str,
        choices=["ensemble", "control", "both"],
        default="both",
        help="Type of IFS data to download",
    )
    parser.add_argument(
        "--date-ranges",
        type=str,
        nargs="+",
        default=[
            "2023-01-02T00:2023-01-08T23",
            "2023-04-02T00:2023-04-08T23",
            "2023-07-02T00:2023-07-08T23",
            "2023-10-02T00:2023-10-08T23",
            "2024-01-02T00:2024-01-08T23",
            "2024-04-02T00:2024-04-08T23",
            "2024-07-02T00:2024-07-08T23",
            "2024-10-02T00:2024-10-08T23",
        ],
        help="Date ranges in format 'YYYY-MM-DDTHH:YYYY-MM-DDTHH'",
    )

    args = parser.parse_args()

    print(f"Starting bulk IFS download to {args.output_dir}")
    print(f"Model: {args.model_name}")
    print(f"Download type: {args.download_type}")
    print(f"Date ranges: {len(args.date_ranges)} periods")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    total_periods = len(args.date_ranges)

    for i, date_range in enumerate(args.date_ranges, 1):
        print(f"\n{'=' * 60}")
        print(f"Processing period {i}/{total_periods}: {date_range}")
        print(f"{'=' * 60}")

        try:
            start_date, end_date = parse_date_range(date_range)

            # Calculate number of days for this period
            num_days = (end_date - start_date).days + 1

            print(f"Start date: {start_date}")
            print(f"End date: {end_date}")
            print(f"Duration: {num_days} days")

            # Download ensemble data
            if args.download_type in ["ensemble", "both"]:
                print("\nDownloading ensemble data...")
                download_ifs_ensemble(
                    args.output_dir, start_date, num_days, args.interval, args.model_name
                )

            # Download control data
            if args.download_type in ["control", "both"]:
                print("\nDownloading control data...")
                download_ifs_control(
                    args.output_dir, start_date, num_days, args.interval, args.model_name
                )

        except Exception as e:
            print(f"ERROR processing {date_range}: {e}")
            continue

    print(f"\n{'=' * 60}")
    print("Bulk download completed!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
