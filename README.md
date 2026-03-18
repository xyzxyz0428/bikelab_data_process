# bike_lab_data_process

**bike_lab_data_process** is an open-source processing toolkit for the **Bike Lab** dataset. It is developed to support reproducible multimodal cycling research and the publication of an open-source dataset paper.

The repository currently includes two released code modules and one additional documented processing component:

1. **raw_data_process** – preprocessing of raw multimodal recordings  
2. **headpose_estimation** – camera calibration, helmet rig calibration, head pose estimation, and result analysis  
3. **lidar_data_process** – LiDAR processing workflow documentation, LiDAR-to-LiDAR calibration preparation, and configuration support for a vendor SDK pipeline  

The goal of this project is to convert raw experimental recordings into structured, analysis-ready outputs while keeping the processing workflow as transparent and reproducible as possible.

---

## Project scope

`bike_lab_data_process` is designed for multimodal data collected on an instrumented bicycle platform. Depending on the experiment, the raw data may include:

- Eyetracker scene camera gaze behavior data and imu data.    
- GNSS data exported from ROS bags  
- LiDAR point clouds  
- Camera images for head pose estimation and timestamped CSV files
- Standalone imu sensor data
- Wheel speed sensor data and powermeter sensor data
- Steering angle data

At the current stage, this project supports:

- extracting and trimming valid data intervals, converting ROS bag data into CSV tables, merging selected sensor outputs into spreadsheet-friendly files  
- estimating head pose from AprilTag-based helmet rigid observations  
- preparing LiDAR-to-LiDAR extrinsic calibration and rostopic generation from pcap file for downstream LiDAR processing ,documenting LiDAR processing configuration files for reproducibility  
---

## Repository structure

Based on the current repository structure on the `main` branch, the top-level folders are:

```text
bikelab_data_process/
├── headpose_estimation/
├── lidar2lidar_calibration/
├── raw_data_process/
└── README.md
```

### Module overview

- `raw_data_process/`  
  Scripts for exporting, trimming, merging, and organizing raw multimodal data.

- `headpose_estimation/`  
  Scripts for intrinsic calibration, helmet rig calibration, frame-based head pose estimation, and result analysis.  

- `lidar_data_preprocess/`  
   Scripts for the estimation of LiDAR-to-LiDAR extrinsic transforms and SDK link for generation of rostopic lidar_data_preprocess for the downstream LiDAR processing pipeline. The resulting calibration can then be converted into YAML or related configuration files, which is required by a proprietary vendor SDK. 

---

## Tested environment

This workflow has been tested with:

- **Ubuntu:** 22.04  
- **Python 3**  
Additional tools used in the workflow include:

- **ffmpeg**  
- common Python scientific packages required by the scripts  

## Additional dependency

Please prepare the ROS 2 bag export utility:

- `ros2_unbag`: <https://github.com/ika-rwth-aachen/ros2_unbag>

This package is used to export selected ROS 2 topics from `.db3` bag files into CSV format.

---

## Installation and preparation

### 1. Clone this repository

```bash
git clone https://github.com/xyzxyz0428/bikelab_data_process.git
cd bikelab_data_process
```

### 2. Prepare `ros2_unbag`

Follow the installation instructions from the upstream repository:

<https://github.com/ika-rwth-aachen/ros2_unbag>

Make sure the command below works in your ROS 2 environment:

```bash
ros2 unbag --help
```

### 3. Prepare Python environment

Depending on your setup, you may use either a system Python installation or a virtual environment. For head pose estimation, a dedicated environment is recommended. Install the required Python packages according to the repository dependency files or the imports used by each script.

---

## Processing pipeline overview

# 1. Raw data processing

This part prepares raw recordings for downstream analysis.

## 1.1 Typical original data files

A typical raw export directory may contain files such as:
```text
steering_angle_20260310_170336.csv
speed_decoded_20260310_170338.csv
rally_payload_decoded_20260310_170337.csv
imu_20260310_170336.csv
rosbag2_2026_03_10-17_10_36/
```
Depending on the experiment and the export stage, additional files may also be present.
These files typically represent:

- `steering_angle_*.csv` – decoded steering angle measurements  
- `speed_decoded_*.csv` – decoded wheel speed or speed sensor output  
- `rally_payload_decoded_*.csv` – decoded payload / interface data  
- `imu_*.csv` – inertial measurement data  
- `rosbag2_*` – raw ROS 2 bag folder containing recorded GNSS topics  
---

## 1.2 Camera data

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

> This step is currently manual or semi-manual, depending on the workflow used for a given dataset release.

---

## 1.3 GPS processing

### Step 1: Export GPS topic from ROS 2 bag

Use `ros2_unbag` to export the desired topic from a ROS 2 `.db3` bag file:

```bash
ros2 unbag /raw_data_process/source/rosbag2_2026_03_10-17_10_36/rosbag2_2026_03_10-17_10_36_0.db3   --output-dir /raw_data_process/source   --naming "%name"   --export /ubx_nav_pvt:table/csv@single_file
```

**Input**
- ROS 2 bag file (`.db3`)  
- topic: `/ubx_nav_pvt`  

**Output**
- CSV export of the selected topic in `exports_csv/`

**Purpose**
- convert ROS-native data into tabular files that are easier to inspect and merge  

---

## 1.4 Merge CSV files into XLSX

### Step 1: Merge selected CSV files over the valid time interval

After selecting the valid time interval, merge relevant CSV files and export them as a single Excel workbook.

```bash
python3 /raw_data_process/script/merge_bikelab_csvs_to_xlsx.py   -i /raw_data_process/source   -o /raw_data_process/result/bike_interface_merged.xlsx   --start-unix-ns 1773159067578250000   --end-unix-ns 1773159563211650000
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

# 2. Head pose estimation

This part covers the full workflow from calibration image collection to final head pose analysis.

## 2.1 Save calibration images from ROS topic stream

### Step 1: Save images for camera calibration

Capture images from a ROS topic at a fixed interval:
on rpi computer2 
```bash
ros2 run camera_streamer camera_publisher
```
on local workstation
```bash
python3 /headpose_estimation/camera_calibration/save_calib_images.py --ros-args -p sec_per_frame:=1.0 -p max_images:=60 -p output_dir:=/headpose_estimation/camera_calibration/calib_images
```

**Output**
- calibration images saved to `/headpose_estimation/camera_calibration/calib_images`

**Purpose**
- collect images for intrinsic camera calibration  

---

## 2.2 Run camera intrinsic calibration

### Step 1: Estimate camera intrinsics with chessboardand generate `camera.json`

```bash
python3 /headpose_estimation/camera_calibration/calibrate_camera_offline.py --image-dir calib_images --cols 5 --rows 7 --square-size-m 0.031 --output-json /headpose_estimation/camera_calibration/camera.json --preview-dir /headpose_estimation/camera_calibration/calib_preview/ --model pinhole
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
- After calibration, copy the estimated intrinsic matrix `k` and distortion parameters `dist` into the final `camera.json` under /headpose_estimation/scripts/camera.json used by the real pipeline if needed.  
- Inspect the reprojection error before accepting the calibration.  

**Output**
- `camera.json`  
- optional preview images in `/headpose_estimation/camera_calibratino/calib_preview/`  

---

## 2.3 Run helmet rig calibration

### Step 1: Estimate the rigid helmet-camera relationship and generate `rig_calib.json`

```bash
python3 /headpose_estimation/scripts/calibrate_helmet_rig.py --camera /headpose_estimation/scripts/camera.json --config /headpose_estimation/scripts/head_rig_config.json --image-dir /headpose_estimation/source/calibration_images --output /headpose_estimation/scripts/rig_calib.json
```

Example output:

```text
tag 1: 20 samples
tag 3: 12 samples
tag 4: 4 samples
tag 5: 5 samples
used_images = 31
saved rig calibration to rig_calib.json
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

## 2.4 Run head pose estimation

### Step 1: Estimate frame-wise head pose and export CSV

```bash
python3 /headpose_estimation/scripts/estimate_headpose_from_frames.py --camera /headpose_estimation/scripts/camera.json --config /headpose_estimation/scripts/head_rig_config.json --rig-calib /headpose_estimation/scripts/rig_calib.json --frame-dir /headpose_estimation/scource/frames_static --timestamps-csv /headpose_estimation/scource/frames_static/timestamps.csv --output-csv /headpose_estimation/result/headpose_output.csv --neutral-frame /headpose_estimation/source/nature.png
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

## 2.5 Analyze head pose results

### Step 1: Run quality analysis on the output CSV

```bash
source ~/venvs/headpose/bin/activate
python3 /headpose_estimation/scripts/analyze_headpose_csv.py   --csv /headpose_estimation/result/headpose_output.csv   --only-ok   --min-head-tags 2   --max-rmse 5
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

# 3. LiDAR data processing

LiDAR data processing in this project relies on a **vendor SDK** that requires a **paid license**.

## 3.1 LiDAR data process preparation

Before running the downstream LiDAR processing pipeline, LiDAR-to-LiDAR extrinsic calibration is performed to prepare calibration results that can later be written into YAML or related configuration files.

### Step 1: Run LiDAR-to-LiDAR calibration

```bash
python3 /lidar2lidar_calibration/script/lidar2lidar_calibration.py   --source_csv /lidar2lidar_calibration/source/indoor/b8.csv   --target_csv /lidar2lidar_calibration/source/indoor/f8.csv   --skip_header   --voxel_size 0.05   --cols 0 1 2
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
- calibration results that can be manually or programmatically converted into LiDAR calibration YAML / config files for the vendor SDK pipeline  

**Purpose**
- align multiple LiDAR sensors into a common reference frame  
- prepare extrinsic parameters for downstream LiDAR processing configuration  
- support fused point cloud processing and multi-sensor spatial consistency  

> A static indoor calibration scene with sufficient geometric structure is recommended.

---

## 3.2 Open-source boundary

Because of licensing restrictions:

- the vendor SDK itself cannot be redistributed in this repository  
- SDK-dependent LiDAR processing code cannot be fully open-sourced here  

However, to support reproducibility, this project will still publish selected non-proprietary configuration files and examples, such as:

- `usr_config`  
- calibration configuration files  
- `lidar_config`  
- selected YAML configuration files used in the processing pipeline  

## 3.3 Notes for users

To reproduce the LiDAR processing workflow, users will need:

1. valid access to the corresponding vendor SDK  
2. an appropriate paid license from the vendor  
3. the configuration files provided in this repository  
4. compatible sensor data and environment setup  

This approach documents the experimental setup and preserves as much reproducibility as possible while respecting third-party licensing constraints.

---

## Inputs and outputs summary

| Module | Main input | Main output |
|---|---|---|
| raw_data_process | video, ROS bag, CSV files | frames, exported CSV, merged XLSX |
| headpose_estimation | images, camera config, rig config, timestamps | `camera.json`, `rig_calib.json`, `headpose_output.csv`,analysis folder|
| lidar_data_process preparation | source/target LiDAR CSV , vendor latest SDK,|  usr configs, YAML configs, rostopic generated,|

---

## Current status

This repository is under active development.

At present, the documented modules are:

- raw data processing  
- head pose estimation  
- LiDAR process preparation  

Additional modules, cleanup, dependency pinning, and example datasets may be added in future releases.

---

## Citation

If you use this repository in your research, please cite the corresponding dataset paper once published.

```bibtex
TDTDTDTDTDTD
```

---

## License

TDTDTDTDTD Please add your intended open-source license here, for example:

- MIT  
- BSD-3-Clause  
- Apache-2.0  

> Note: third-party vendor SDK components are **not** covered by the repository license and must be obtained separately from the vendor.

---

## Contact

For questions, issues, or collaboration requests, please open an issue in this repository or contact the maintainers. 
Xinyu Zhang (xinyu.zhang@tu-dresden.de)
