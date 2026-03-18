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

- camera videos / image streams  
- GNSS / GPS data exported from ROS bags  
- LiDAR point clouds  
- head-mounted camera images for head pose estimation  
- timestamped CSV files from different acquisition pipelines  
- decoded bicycle interface data such as steering angle, speed, IMU, and payload-related signals  

At the current stage, this project supports:

- extracting and trimming valid data intervals  
- converting ROS bag data into CSV tables  
- merging selected sensor outputs into spreadsheet-friendly files  
- preparing LiDAR-to-LiDAR extrinsic calibration for downstream LiDAR processing  
- estimating head pose from AprilTag-based helmet rig observations  
- documenting LiDAR processing configuration files for reproducibility  

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

- `lidar2lidar_calibration/`  
  Scripts used during **LiDAR data process preparation** to estimate LiDAR-to-LiDAR extrinsic transforms. The resulting calibration can then be converted into YAML or related configuration files for the downstream LiDAR processing pipeline.

- `lidar_data_process/`  
  This workflow is documented in this README, but it is **not currently released as a full open-source code folder** in this repository because part of the LiDAR pipeline depends on a proprietary vendor SDK.

---

## Tested environment

This workflow has been tested with:

- **ROS 2:** Humble  
- **Ubuntu:** 22.04  

Additional tools used in the workflow include:

- **Python 3**  
- **ffmpeg**  
- common Python scientific packages required by the scripts  
- an optional Python virtual environment for head pose estimation  

---

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

Depending on your setup, you may use either a system Python installation or a virtual environment. For head pose estimation, a dedicated environment is recommended.

Example:

```bash
python3 -m venv ~/venvs/headpose
source ~/venvs/headpose/bin/activate
```

Install the required Python packages according to the repository dependency files or the imports used by each script.

---

## Processing pipeline overview

The current workflow is organized into three parts:

1. **Raw data processing**  
2. **Head pose estimation**  
3. **LiDAR data processing**  

A typical processing sequence is:

1. export or prepare raw sensor data  
2. determine the valid time interval for a recording  
3. convert data into analysis-ready intermediate files  
4. run calibration steps  
5. estimate target quantities such as head pose  
6. prepare LiDAR extrinsic calibration and configuration files  
7. validate outputs and inspect quality metrics  

---

# 1. Raw data processing

This part prepares raw recordings for downstream analysis.

## 1.1 Typical original data files

A typical raw export directory may contain files such as:

```text
ubx_nav_pvt.csv
steering_angle_20260310_170336.csv
speed_decoded_20260310_170338.csv
rally_payload_decoded_20260310_170337.csv
imu_20260310_170336.csv
rosbag2_2026_03_10-17_10_36/
```

These files typically represent:

- `ubx_nav_pvt.csv` – GNSS navigation solution exported from ROS bag data  
- `steering_angle_*.csv` – decoded steering angle measurements  
- `speed_decoded_*.csv` – decoded wheel speed or speed sensor output  
- `rally_payload_decoded_*.csv` – decoded payload / interface data  
- `imu_*.csv` – inertial measurement data  
- `rosbag2_*` – raw ROS 2 bag folder containing recorded topics  

Depending on the experiment and the export stage, additional files may also be present.

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
ros2 unbag /mnt/bikelab_data/IB_Lab/lidar/20260226_164803/20260226_164803_0.db3   --output-dir /mnt/bikelab_data/IB_Lab/lidar/20260226_164803/exports_csv   --naming "%name"   --export /ubx_nav_pvt:table/csv@single_file
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
python3 merge_bikelab_csvs_to_xlsx.py   -i /mnt/bikelab_data/IB_Lab/bike_interface_data/20260310   -o bike_interface_merged.xlsx   --start-unix-ns 1773159067578250000   --end-unix-ns 1773159563211650000
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

```bash
python3 save_calib_images.py --ros-args -p sec_per_frame:=1.0 -p max_images:=60 -p output_dir:=/home/pi/calib_images
```

**Output**
- calibration images saved to `/home/pi/calib_images`

**Purpose**
- collect images for intrinsic camera calibration  

---

## 2.2 Run camera intrinsic calibration

### Step 1: Estimate camera intrinsics and generate `camera.json`

```bash
python3 calibrate_camera_offline.py --image-dir calib_images --cols 5 --rows 7 --square-size-m 0.031 --output-json camera.json --preview-dir calib_preview --model pinhole
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

## 2.3 Run helmet rig calibration

### Step 1: Estimate the rigid helmet-camera relationship and generate `rig_calib.json`

```bash
python3 calibrate_helmet_rig.py --camera camera.json --config head_rig_config.json --image-dir calibration_images --output rig_calib.json
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
python3 estimate_headpose_from_frames.py --camera camera.json --config head_rig_config.json --rig-calib rig_calib.json --frame-dir frames_static --timestamps-csv frames_static/timestamps.csv --output-csv headpose_output.csv --neutral-frame nature.png
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
python3 analyze_headpose_csv.py   --csv headpose_output.csv   --only-ok   --min-head-tags 2   --max-rmse 5
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
python3 lidar2lidar_calibration.py   --source_csv indoor/b8.csv   --target_csv indoor/f8.csv   --skip_header   --voxel_size 0.05   --cols 0 1 2
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
| headpose_estimation | images, camera config, rig config, timestamps | `camera.json`, `rig_calib.json`, `headpose_output.csv` |
| lidar_data_process preparation | source/target LiDAR CSV | rigid transform and calibration values for YAML / config generation |
| lidar_data_process | vendor SDK, LiDAR data, YAML configs | processed LiDAR outputs depending on the licensed pipeline |

---

## Recommended workflow for dataset publication

For each recording session, a typical reproducible workflow is:

1. extract camera frames from raw video  
2. identify the valid time interval  
3. trim frames and timestamps accordingly  
4. export GPS or other ROS topics from bag files  
5. merge selected CSV outputs into structured tables  
6. calibrate the camera and helmet rig  
7. estimate head pose  
8. analyze quality metrics and retain valid results only  
9. run LiDAR-to-LiDAR calibration as preparation for LiDAR configuration  
10. generate or document LiDAR calibration YAML and processing settings  

This makes it easier to prepare consistent public releases for a dataset paper.

---

## Reproducibility notes

To improve reproducibility, we recommend storing the following for each processed sequence:

- raw recording identifier  
- start and end timestamps used for valid interval selection  
- calibration files (`camera.json`, `rig_calib.json`)  
- LiDAR processing configuration files  
- LiDAR extrinsic calibration results used to generate YAML or config files  
- script version or commit hash  
- output quality metrics  
- notes on failed frames or excluded segments  

---

## Current status

This repository is under active development.

At present, the documented modules are:

- raw data processing  
- head pose estimation  
- LiDAR data processing configuration support  
- LiDAR-to-LiDAR calibration for LiDAR process preparation  

Additional modules, cleanup, dependency pinning, and example datasets may be added in future releases.

---

## Citation

If you use this repository in your research, please cite the corresponding dataset paper once published.

```bibtex
@misc{bike_lab_data_process,
  title        = {bike_lab_data_process: Processing tools for the Bike Lab dataset},
  author       = {Author names to be added},
  year         = {2026},
  howpublished = {GitHub repository},
  note         = {Associated with the Bike Lab open-source dataset publication}
}
```

---

## License

Please add your intended open-source license here, for example:

- MIT  
- BSD-3-Clause  
- Apache-2.0  

> Note: third-party vendor SDK components are **not** covered by the repository license and must be obtained separately from the vendor.

---

## Contact

For questions, issues, or collaboration requests, please open an issue in this repository or contact the maintainers.
