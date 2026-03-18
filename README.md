# bike_lab_data_process

Open-source processing toolkit for the **Bike Lab** dataset, developed to support reproducible multimodal cycling research and dataset publication.

This repository currently contains four main parts:

1. **raw_data_process** – preprocessing of raw multimodal recordings
2. **lidar2lidar_calibration** – rigid calibration between LiDAR sensors
3. **headpose_estimation** – camera calibration, helmet rig calibration, head pose estimation, and result analysis
4. **lidar_data_process** – LiDAR data processing based on vendor SDK and configuration files

The project is intended to support dataset preparation and validation for an open-source dataset paper. It focuses on converting raw recordings into structured, analysis-ready data products.

---

## Project scope

`bike_lab_data_process` is designed for multimodal data collected on an instrumented bicycle platform. Depending on the experiment, the raw data may include:

- camera videos / image streams
- GNSS / GPS data exported from ROS bags
- LiDAR point clouds
- head-mounted camera images for head pose estimation
- timestamped CSV files from different acquisition pipelines

At the current stage, this repository provides the essential processing steps needed for:

- extracting and trimming valid data intervals
- converting ROS bag data into CSV tables
- merging selected sensor outputs into spreadsheet-friendly files
- calibrating LiDAR-to-LiDAR extrinsics
- estimating head pose from AprilTag-based helmet rig observations
- documenting LiDAR processing configuration files for reproducibility

---

## Repository structure

```text
bike_lab_data_process/
├── raw_data_process/
│   ├── video frame extraction
│   ├── GPS / ROS bag export
│   └── CSV merging and XLSX export
├── lidar2lidar_calibration/
│   └── LiDAR extrinsic calibration scripts
├── headpose_estimation/
│   ├── image capture from ROS topics
│   ├── camera intrinsic calibration
│   ├── helmet rig calibration
│   ├── head pose estimation
│   └── result analysis
├── lidar_data_process/
│   ├── vendor SDK related processing
│   ├── published YAML configuration examples
│   └── sensor setup and calibration configuration
└── README.md
```

> The exact internal file organization may evolve as the repository matures. This README documents the current processing workflow.

---

## Tested environment

This workflow has been tested with:

- **ROS 2 version:** Humble
- **Ubuntu version:** 22.04

Additional tools used in the workflow:

- **Python 3**
- **ffmpeg**
- common Python scientific packages used by the scripts
- optional Python virtual environment for head pose estimation

---

## Additional dependency

Please prepare the ROS 2 bag export utility:

- `ros2_unbag`: <https://github.com/ika-rwth-aachen/ros2_unbag>

This package is used to export selected ROS 2 topics from `.db3` bag files into CSV format.

---

## Installation and preparation

### 1. Clone this repository

```bash
git clone <your-repository-url>
cd bike_lab_data_process
```

### 2. Prepare `ros2_unbag`

Follow the installation instructions from the upstream repository:

<https://github.com/ika-rwth-aachen/ros2_unbag>

Make sure the command below works in your ROS 2 environment:

```bash
ros2 unbag --help
```

### 3. Prepare Python environment

Depending on your setup, you may use either a system Python or a virtual environment. For head pose estimation, a dedicated environment is recommended.

Example:

```bash
python3 -m venv ~/venvs/headpose
source ~/venvs/headpose/bin/activate
```

Install the packages required by your scripts according to the repository's dependency files or script imports.

---

## Processing pipeline overview

The current workflow is organized into four parts:

1. **Raw data processing**
2. **LiDAR-to-LiDAR calibration**
3. **Head pose estimation**
4. **LiDAR data processing**

A typical processing sequence is:

1. export or prepare raw sensor data
2. determine the valid time interval for a recording
3. convert data into analysis-ready intermediate files
4. run calibration steps
5. estimate target quantities such as head pose
6. validate outputs and inspect quality metrics

---

# 1. Raw data processing

This part prepares raw recordings for downstream analysis.

## 1.1 Camera data

### Step 1: Convert video to image frames

Use `ffmpeg` to extract frames from a video file:

```bash
ffmpeg -i video.avi frames/frame_%06d.png
```

**Input**
- raw video file, e.g. `video.avi`

**Output**
- extracted image sequence in the `frames/` directory

### Step 2: Determine valid start and end time

After frame extraction, identify the valid temporal interval of the recording.

Then:

- delete invalid frames outside the selected interval
- remove corresponding rows from the timestamp CSV file

**Purpose**
- keep only the synchronized and valid portion of the dataset
- ensure consistent downstream processing

> This step is currently manual or semi-manual depending on your workflow.

---

## 1.2 GPS processing

### Export GPS topic from ROS 2 bag

Use `ros2_unbag` to export the desired topic from a ROS 2 `.db3` bag file:

```bash
ros2 unbag /mnt/bikelab_data/IB_Lab/lidar/20260226_164803/20260226_164803_0.db3 \
  --output-dir /mnt/bikelab_data/IB_Lab/lidar/20260226_164803/exports_csv \
  --naming "%name" \
  --export /ubx_nav_pvt:table/csv@single_file
```

**Input**
- ROS 2 bag file (`.db3`)
- topic: `/ubx_nav_pvt`

**Output**
- CSV export of the selected topic in `exports_csv/`

**Purpose**
- convert ROS-native data into tabular files that are easier to inspect and merge

---

## 1.3 Merge CSV files into XLSX

After selecting the valid time interval, merge relevant CSV files and export them as a single Excel workbook.

```bash
python3 merge_bikelab_csvs_to_xlsx.py \
  -i /mnt/bikelab_data/IB_Lab/bike_interface_data/20260310 \
  -o bike_interface_merged.xlsx \
  --start-unix-ns 1773159067578250000 \
  --end-unix-ns 1773159563211650000
```

**Input**
- folder containing source CSV files
- valid start and end timestamps in Unix nanoseconds

**Output**
- merged spreadsheet file, e.g. `bike_interface_merged.xlsx`

**Typical operations**
- extract only the valid time interval
- keep selected columns
- export a compact spreadsheet for inspection or annotation

---

# 2. LiDAR-to-LiDAR calibration

This part estimates the rigid transformation between two LiDAR sensors.

## Run calibration

```bash
python3 lidar2lidar_calibration.py \
  --source_csv indoor/b8.csv \
  --target_csv indoor/f8.csv \
  --skip_header \
  --voxel_size 0.05 \
  --cols 0 1 2
```

**Input**
- source point cloud CSV file
- target point cloud CSV file
- selected columns representing XYZ coordinates

**Important arguments**
- `--source_csv`: source LiDAR point cloud
- `--target_csv`: target LiDAR point cloud
- `--skip_header`: skip CSV header row if present
- `--voxel_size 0.05`: downsampling voxel size used during registration
- `--cols 0 1 2`: use columns 0, 1, 2 as XYZ

**Output**
- estimated rigid transformation from source LiDAR frame to target LiDAR frame
- depending on script implementation, this may include console output, saved transform parameters, or visualization

**Purpose**
- align multiple LiDAR sensors into a common reference frame
- support fused point cloud processing and multi-sensor spatial consistency

> It is recommended to use static indoor calibration scenes with sufficient geometric structure.

---

# 3. Head pose estimation

This part covers the full workflow from calibration image collection to final head pose analysis.

## 3.1 Save calibration images from ROS topic stream

Capture images from a ROS topic at a fixed interval:

```bash
python3 save_calib_images.py \
  --ros-args -p sec_per_frame:=1.0 -p max_images:=60 -p output_dir:=/home/pi/calib_images
```

**Output**
- calibration images saved to `/home/pi/calib_images`

**Purpose**
- collect images for intrinsic camera calibration

---

## 3.2 Run camera intrinsic calibration

Estimate camera intrinsics and save them to `camera.json`:

```bash
python3 calibrate_camera_offline.py \
  --image-dir calib_images \
  --cols 5 \
  --rows 7 \
  --square-size-m 0.031 \
  --output-json camera.json \
  --preview-dir calib_preview \
  --model pinhole
```

Example output:

```text
=== Calibration done ===
Model: pinhole
Image size: 640 x 480
Valid images: 54 / 60
Calibration RMS: 3.847388
Reprojection RMSE: 3.847388 px
Saved to: camera.json
```

**Notes**
- After calibration, copy the estimated intrinsic matrix `k` and distortion parameters `dist` into the final `camera.json` used by the real pipeline if needed.
- Inspect the reprojection error before accepting the calibration.

**Output**
- `camera.json`
- optional preview images in `calib_preview/`

---

## 3.3 Run helmet rig calibration

Estimate the rigid relationship between the camera and helmet-mounted AprilTags:

```bash
python3 calibrate_helmet_rig.py \
  --camera camera.json \
  --config head_rig_config.json \
  --image-dir calibration_images \
  --output rig_calib.json
```

**Input**
- calibrated `camera.json`
- helmet/tag configuration file `head_rig_config.json`
- calibration image folder

**Output**
- `rig_calib.json`

**Purpose**
- estimate the fixed camera-to-helmet rigid transform used later for head pose estimation

---

## 3.4 Run head pose estimation

Estimate head pose from extracted image frames:

```bash
python3 estimate_headpose_from_frames.py \
  --camera camera.json \
  --config head_rig_config.json \
  --rig-calib rig_calib.json \
  --frame-dir frames_static \
  --timestamps-csv frames_static/timestamps.csv \
  --output-csv headpose_output.csv \
  --neutral-frame nature.png
```

Example output:

```text
saved to headpose_output.csv
```

**Input**
- `camera.json`
- `head_rig_config.json`
- `rig_calib.json`
- image frames directory
- timestamps CSV file
- neutral frame image

**Output**
- `headpose_output.csv`

**Purpose**
- estimate per-frame head pose from the calibrated camera and helmet rig setup

---

## 3.5 Analyze head pose results

Activate the head pose environment and analyze the output CSV:

```bash
source ~/venvs/headpose/bin/activate

python3 analyze_headpose_csv.py \
  --csv headpose_output.csv \
  --only-ok \
  --min-head-tags 2 \
  --max-rmse 5
```

**Purpose**
- filter valid estimates
- check reconstruction quality
- inspect whether the head pose estimates satisfy quality thresholds

**Recommended criteria**
- only keep successful estimates
- require at least 2 detected head tags
- reject frames with RMSE above 5 px

---

# 4. LiDAR data processing

LiDAR data processing in this project relies on a **vendor SDK** that requires a **paid license**.

## Open-source boundary

Because of licensing restrictions:

- the vendor SDK itself cannot be redistributed in this repository
- SDK-dependent LiDAR processing code cannot be fully open-sourced here

However, to support reproducibility, this repository will still publish selected non-proprietary configuration files and examples, such as:

- `user config`
- `calibration config`
- `lidar_config`
- selected YAML configuration files used in the processing pipeline

## Notes for users

To reproduce the LiDAR processing workflow, users will need:

1. valid access to the corresponding vendor SDK
2. an appropriate paid license from the vendor
3. the configuration files provided in this repository
4. compatible sensor data and environment setup

This design allows us to document the experimental setup and preserve as much reproducibility as possible while respecting third-party licensing constraints.

---

## Inputs and outputs summary

| Module | Main input | Main output |
|---|---|---|
| raw_data_process | video, ROS bag, CSV files | frames, exported CSV, merged XLSX |
| lidar2lidar_calibration | source/target LiDAR CSV | rigid transform / calibration result |
| headpose_estimation | images, camera config, rig config, timestamps | `camera.json`, `rig_calib.json`, `headpose_output.csv` |
| lidar_data_process | vendor SDK, LiDAR data, YAML configs | processed LiDAR outputs depending on licensed pipeline |

---

## Recommended workflow for dataset publication

For each recording session, a typical reproducible workflow is:

1. extract camera frames from raw video
2. identify the valid time interval
3. trim frames and timestamps accordingly
4. export GPS or other ROS topics from bag files
5. merge selected CSV outputs into structured tables
6. run LiDAR extrinsic calibration when needed
7. calibrate the camera and helmet rig
8. estimate head pose
9. analyze quality metrics and retain valid results only
10. document LiDAR SDK configuration and processing settings

This makes it easier to prepare consistent public releases for a dataset paper.

---

## Reproducibility notes

To improve reproducibility, we recommend storing the following for each processed sequence:

- raw recording identifier
- start and end timestamps used for valid interval selection
- calibration files (`camera.json`, `rig_calib.json`)
- LiDAR processing configuration files
- script version or commit hash
- output quality metrics
- notes on failed frames or excluded segments

---

## Current status

This repository is under active development.

At present, the documented modules are:

- raw data processing
- LiDAR-to-LiDAR calibration
- head pose estimation
- LiDAR data processing configuration support

Additional modules, cleanup, dependency pinning, and example datasets may be added in future releases.

---

## Citation

If you use this repository in your research, please cite the corresponding dataset paper once published.

```bibtex
@misc{bike_lab_data_process,
  title        = {A mutimodal instrumented bicycle platform and dataset capturing urban environments, cyclis gaze,rider inputs, and bicycle dynamics},
  author       = {Xinyu Zhang, Yikai Zeng, Meng Wang},
  year         = {2026},
  howpublished = {GitHub repository},
  note         = {Associated with the Bike Lab open-source dataset publication}
}
```
---

## Contact

For questions, issues, or collaboration requests, please open an issue in this repository or contact the maintainers.
