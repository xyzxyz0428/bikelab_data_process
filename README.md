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

> This step is currently manual or semi-manual, depending on the workflow used for a given dataset release

---

## 1.3 lidar data

### Step 1: Convert pcap to csv for timestamp exctraction

Use `tshrk` to extract timestamp:

```bash
/raw_data_process/source/extract_lidar_packet_timestamps.sh your_capture.pcap lidar_packet_csvs \
    192.168.1.200 192.168.1.201 192.168.1.202
```

**Input**
- your_capture.pcap

**Output**
- extracted csv files in "lidar_packet_csvs \" directory

---

## 1.4 GPS processing

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

## 1.5 Merge CSV files into XLSX

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
## 1.6 Analysis
Run the validation suite on the merged XLSX to generate dataset-paper figures and summary tables, including stream health, timing consistency, LiDAR frame health, and optional behavioral sanity-check plots.
```bash
python raw_data_process/script/dataset_paper_validation_suite_v3.py \
  --xlsx raw_data_process/result/bike_interface_merged_with_lidar.xlsx \
  --outdir raw_data_process/result/validation_outputs
```

# 2. Head pose estimation

This part covers the full workflow from camera calibration to world-level gaze evaluation.

---

## 2.1 Save calibration images

### Step 1: Start the camera publisher

On the Raspberry Pi:

```bash id="9qpt9x"
ros2 run camera_streamer camera_publisher
```

### Step 2: Save calibration images

On the local workstation:

```bash id="c19m4v"
python3 /headpose_estimation/camera_calibration/save_calib_images.py \
  --ros-args \
  -p sec_per_frame:=1.0 \
  -p max_images:=60 \
  -p output_dir:=/headpose_estimation/camera_calibration/calib_images
```

**Output**

* calibration images in `calib_images`

**Purpose**

* collect images for intrinsic camera calibration

---

## 2.2 Run camera intrinsic calibration

### Step 1: Generate `camera.json`

```bash id="ldl1dw"
python3 /headpose_estimation/camera_calibration/calibrate_camera_offline.py \
  --image-dir /headpose_estimation/camera_calibration/calib_images \
  --cols 5 \
  --rows 7 \
  --square-size-m 0.031 \
  --output-json /headpose_estimation/camera_calibration/camera.json \
  --preview-dir /headpose_estimation/camera_calibration/calib_preview \
  --model pinhole
```

Example output:

```text id="ut8tdz"
=== Calibration done ===
Model: pinhole
Image size: 640 x 480
Valid images: 54 / 60
Calibration RMS: 3.847388
Reprojection RMSE: 3.847388 px
Saved to: camera.json
```

**Output**

* `camera.json`
* preview images in `calib_preview`

**Purpose**

* estimate camera intrinsics for the back camera

**Note**

* inspect reprojection error before using the result

---

## 2.3 Run helmet rig calibration

### Step 1: Generate `rig_calib.json`

```bash id="1ji56l"
python3 /headpose_estimation/scripts/calibrate_helmet_rig.py \
  --camera /headpose_estimation/scripts/camera.json \
  --config /headpose_estimation/scripts/head_rig_config.json \
  --image-dir /headpose_estimation/source/calibration_images \
  --output /headpose_estimation/scripts/rig_calib.json
```

Example output:

```text id="ilyv67"
tag 1: 20 samples
tag 3: 12 samples
tag 4: 4 samples
tag 5: 5 samples
used_images = 31
saved rig calibration to rig_calib.json
```

**Output**

* `rig_calib.json`

**Purpose**

* estimate the fixed geometry of the helmet tag rig

---

## 2.4 Run head pose estimation

### Step 1: Generate `headpose_output.csv`

```bash id="mtdhfh"
python3 /headpose_estimation/scripts/estimate_headpose_from_frames.py \
  --camera /headpose_estimation/scripts/camera.json \
  --config /headpose_estimation/scripts/head_rig_config.json \
  --rig-calib /headpose_estimation/scripts/rig_calib.json \
  --frame-dir /headpose_estimation/source/frames_static \
  --timestamps-csv /headpose_estimation/source/frames_static/timestamps.csv \
  --output-csv /headpose_estimation/result/headpose_output.csv \
  --neutral-frame /headpose_estimation/source/neutral.png \
  --min-head-tags 2 \
  --max-head-rmse-px 5
```

**Output**

* `headpose_output.csv`

**Purpose**

* estimate frame-wise head pose
* export per-frame quality fields:

  * `num_head_tags`
  * `visible_head_tag_ids`
  * `head_rmse_px`
  * `head_quality_ok`

---

## 2.5 Analyze head pose results

### Step 1: Run head pose quality analysis

```bash id="u5hcyw"
source ~/venvs/headpose/bin/activate
python3 /headpose_estimation/scripts/analyze_headpose_csv.py \
  --csv /headpose_estimation/result/headpose_output.csv \
  --only-ok \
  --only-head-quality-ok \
  --min-head-tags 2 \
  --max-rmse 2 \
  --out-dir /headpose_estimation/result/headpose_analysis
```

**Output**

* filtered CSV
* angle plots
* RMSE plots

**Purpose**

* filter valid head pose frames
* inspect head pose stability and reconstruction quality

**Recommended criteria**

* keep only `ok == 1`
* keep only `head_quality_ok == 1`
* require at least 2 visible head tags
* reject frames with RMSE above 5 px

---

## 2.6 Build AprilTag world baseline

### Step 1: Generate `apriltag_baseline.json`

```bash id="3m2ww7"
python3 /headpose_estimation/scripts/build_apriltag_baseline_from_back_camera.py \
  --camera-json /headpose_estimation/scripts/camera.json \
  --frame-dir /headpose_estimation/source/baseline_frames \
  --timestamps-csv /headpose_estimation/source/baseline_timestamps.csv \
  --tag-family tag36h11 \
  --default-size-m 0.10 \
  --ref-tag-id 17 \
  --target-tag-ids 22,23,24,25,26,15,16,17,19,20,9,10,11,12,13,14 \
  --min-samples 10 \
  --max-translation-std-m 0.01 \
  --max-rotation-std-deg 2.0 \
  --output-json /headpose_estimation/result/apriltag_baseline.json
```

**Output**

* `apriltag_baseline.json`

**Purpose**

* define world frame `W`
* estimate world positions of target tags
* export tag quality:

  * `translation_std_m`
  * `rotation_std_deg`
  * `low_confidence`

---

## 2.7 Extract scene frame timestamps

### Step 1: Generate `scene_timestamps.csv`

```bash id="0n5t6t"
python3 /headpose_estimation/scripts/generate_scene_frame_timestamps.py \
  --recording-g3 /headpose_estimation/source/tobii_recording/recording.g3 \
  --scene-video /headpose_estimation/source/tobii_recording/scenevideo.mp4 \
  --out-csv /headpose_estimation/result/scene_timestamps.csv \
  --out-dir /headpose_estimation/source/scene_frames
```

**Output**

* `scene_timestamps.csv`
* extracted scene frames

**Purpose**

* assign timestamps to Tobii scene frames
* align scene frames with other data streams

---

## 2.8 Extract target-tag windows

### Step 1: Generate `tag_time_windows.csv`

```bash id="rflfjh"
python3 /headpose_estimation/scripts/extract_tag_time_windows.py \
  --scene-timestamps-csv /headpose_estimation/result/scene_timestamps.csv \
  --output-csv /headpose_estimation/result/tag_time_windows.csv
```

**Output**

* `tag_time_windows.csv`

**Purpose**

* split target tags into continuous evaluation windows

---

## 2.9 Validate Tobii 2D gaze

### Step 1: Validate raw gaze

```bash id="asjlwm"
python3 /headpose_estimation/scripts/validate_tobii_2d_with_tag_windows_v2.py \
  --tag-windows-csv /headpose_estimation/result/tag_time_windows.csv \
  --scene-timestamps-csv /headpose_estimation/result/scene_timestamps.csv \
  --scene-frame-dir /headpose_estimation/source/scene_frames \
  --tobii-xlsx /headpose_estimation/source/tobii_raw.xlsx \
  --recording-g3 /headpose_estimation/source/tobii_recording/recording.g3 \
  --mode raw \
  --window-fraction-start 0.3 \
  --window-fraction-end 0.7 \
  --apriltag-baseline-json /headpose_estimation/result/apriltag_baseline.json \
  --exclude-low-confidence-tags \
  --max-tag-translation-std-m 0.01 \
  --max-tag-rotation-std-deg 2.0 \
  --output-csv /headpose_estimation/result/tobii_2d_validation_raw.csv
```

### Step 2: Validate fixation gaze

```bash id="w4f4qc"
python3 /headpose_estimation/scripts/validate_tobii_2d_with_tag_windows_v2.py \
  --tag-windows-csv /headpose_estimation/result/tag_time_windows.csv \
  --scene-timestamps-csv /headpose_estimation/result/scene_timestamps.csv \
  --scene-frame-dir /headpose_estimation/source/scene_frames \
  --tobii-xlsx /headpose_estimation/source/tobii_fixation.xlsx \
  --recording-g3 /headpose_estimation/source/tobii_recording/recording.g3 \
  --mode fixation \
  --window-fraction-start 0.3 \
  --window-fraction-end 0.7 \
  --apriltag-baseline-json /headpose_estimation/result/apriltag_baseline.json \
  --exclude-low-confidence-tags \
  --max-tag-translation-std-m 0.01 \
  --max-tag-rotation-std-deg 2.0 \
  --output-csv /headpose_estimation/result/tobii_2d_validation_fixation.csv
```

**Output**

* per-frame CSV
* summary CSV

**Purpose**

* validate Tobii gaze in image space
* report:

  * `inside_tag_polygon`
  * `distance_to_polygon_px`
  * normalized error by tag width

---

## 2.10 Estimate `T_H_C1`

### Step 1: Generate `T_H_C1.json`

```bash id="ffweme"
python3 headpose_estimation/scripts/calibrate_T_H_C1_via_common_board.py \
  --back-camera-json headpose_estimation/scripts/camera.json \
  --back-frame-dir new_experiment/01_T_H_C1_calibration/back_frames \
  --back-timestamps-csv new_experiment/01_T_H_C1_calibration/back_timestamps.csv \
  --scene-camera-json headpose_estimation/scripts/scene_camera.json \
  --scene-frame-dir new_experiment/01_T_H_C1_calibration/scene_frames \
  --scene-timestamps-csv new_experiment/01_T_H_C1_calibration/scene_timestamps.csv \
  --rig-calib-json headpose_estimation/scripts/rig_calib.json \
  --board-tag-ids 0,1,2,3,4,5,6,7,8,9,10,11 \
  --board-rows 4 \
  --board-cols 3 \
  --board-tag-size-m 0.040 \
  --board-gap-x-m 0.020 \
  --board-gap-y-m 0.020 \
  --tag-family tag36h11 \
  --min-board-tags-scene 2 \
  --min-board-tags-back 2 \
  --min-head-tags 2 \
  --max-pair-dt-ms 30 \
  --output-json new_experiment/result/T_H_C1.json
```

**Output**

* `T_H_C1.json`

**Purpose**

* estimate the rigid transform from head frame to Tobii scene camera

**Note**

* use near-range common-board data for best accuracy

---

## 2.11 Estimate `T_W_C2`

### Step 1: Generate `T_W_C2.json`

```bash id="mp2qbx"
python3 /headpose_estimation/scripts/estimate_T_W_C2_from_ref17.py \
  --camera-json /headpose_estimation/scripts/camera.json \
  --frame-dir /headpose_estimation/source/baseline_frames \
  --timestamps-csv /headpose_estimation/source/baseline_timestamps.csv \
  --ref-tag-id 17 \
  --tag-size-m 0.10 \
  --min-samples 20 \
  --max-translation-std-m 0.01 \
  --max-rotation-std-deg 2.0 \
  --output-json /headpose_estimation/result/T_W_C2.json
```

**Output**

* `T_W_C2.json`

**Purpose**

* estimate back-camera pose in world frame
* export transform quality:

  * `translation_std_m`
  * `rotation_std_deg`
  * `low_confidence`

---

## 2.12 Estimate `T_C1_HUCS` and build transforms

### Step 1: Generate `T_C1_HUCS.json`

```bash id="1l841y"
python3 /headpose_estimation/scripts/estimate_T_C1_HUCS_from_tobii_2d3d.py \
  --tobii-xlsx /headpose_estimation/source/tobii_raw.xlsx \
  --recording-g3 /headpose_estimation/source/tobii_recording/recording.g3 \
  --scene-camera-json /headpose_estimation/scripts/scene_camera.json \
  --output-json /headpose_estimation/result/T_C1_HUCS.json
```

### Step 2: Generate `transforms.json`

```bash id="x5zsn5"
python3 /headpose_estimation/scripts/make_transforms_json.py \
  --T-W-C2-json /headpose_estimation/result/T_W_C2.json \
  --T-H-C1-json /headpose_estimation/result/T_H_C1.json \
  --T-C1-HUCS-json /headpose_estimation/result/T_C1_HUCS.json \
  --output-json /headpose_estimation/result/transforms.json
```

**Output**

* `T_C1_HUCS.json`
* `transforms.json`

**Purpose**

* assemble all transforms for world-level gaze evaluation
* keep transform metadata in one file

---

## 2.13 Run A/B/C gaze evaluation

### Step 1: Generate `gaze_abc_eval.csv`

```bash id="o69iej"
python3 /headpose_estimation/scripts/evaluate_gaze_abc_by_windows.py \
  --tag-windows-csv /headpose_estimation/result/tag_time_windows.csv \
  --apriltag-baseline-json /headpose_estimation/result/apriltag_baseline.json \
  --headpose-csv /headpose_estimation/result/headpose_output.csv \
  --scene-camera-json /headpose_estimation/scripts/scene_camera.json \
  --transforms-json /headpose_estimation/result/transforms.json \
  --tobii-raw-xlsx /headpose_estimation/source/tobii_raw.xlsx \
  --recording-g3 /headpose_estimation/source/tobii_recording/recording.g3 \
  --window-fraction-start 0.3 \
  --window-fraction-end 0.7 \
  --max-sync-dt-ms 20 \
  --exclude-low-confidence-tags \
  --max-tag-translation-std-m 0.01 \
  --max-tag-rotation-std-deg 2.0 \
  --min-head-tags 2 \
  --max-head-rmse-px 5 \
  --output-csv /headpose_estimation/result/gaze_abc_eval.csv
```

Optional strict B mode:

```bash id="v0ayk7"
python3 /headpose_estimation/scripts/evaluate_gaze_abc_by_windows.py \
  --tag-windows-csv /headpose_estimation/result/tag_time_windows.csv \
  --apriltag-baseline-json /headpose_estimation/result/apriltag_baseline.json \
  --headpose-csv /headpose_estimation/result/headpose_output.csv \
  --scene-camera-json /headpose_estimation/scripts/scene_camera.json \
  --transforms-json /headpose_estimation/result/transforms.json \
  --tobii-raw-xlsx /headpose_estimation/source/tobii_raw.xlsx \
  --recording-g3 /headpose_estimation/source/tobii_recording/recording.g3 \
  --window-fraction-start 0.3 \
  --window-fraction-end 0.7 \
  --max-sync-dt-ms 20 \
  --exclude-low-confidence-tags \
  --max-tag-translation-std-m 0.01 \
  --max-tag-rotation-std-deg 2.0 \
  --min-head-tags 2 \
  --max-head-rmse-px 5 \
  --b-require-both-eyes-valid \
  --output-csv /headpose_estimation/result/gaze_abc_eval_strictB.csv
```

**Methods**

* A: Tobii `Gaze point 3D`
* B: Tobii native 3D gaze ray
* C: 2D gaze + head pose reconstruction

**Output**

* per-row CSV
* summary CSV

**Purpose**

* evaluate gaze in world coordinates
* compare native and reconstructed gaze formulations

---

## 2.14 Review results

Check:

* `tobii_2d_validation_*_summary.csv`
* `gaze_abc_eval_summary.csv`

Focus on:

* image-space gaze quality
* A/B/C mean / median / p95
* `headpose_dt_ms`
* B valid sample count
* head pose quality:

  * `num_head_tags`
  * `head_rmse_px`

---

## 2.15 Recommended next data collection

* re-record `T_H_C1` with a **near-range common board**
* use a **small fixation point** at the target center
* keep each target for **2–3 s**
* record:

  * Tobii raw
  * Tobii fixation
  * back camera frames
  * scene frames
  * head pose
  * baseline tags

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
### export and save perception result

For exporting the RViz perception topic from a ROS1 bag file to Excel in timestamp order, use:

```bash
python3 export_markerarray_to_excel.py your_file.bag --topic /perception_info_rviz --output perception_info_rviz.xlsx   --max-rows 1000000
```
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
