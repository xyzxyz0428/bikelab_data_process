# BikeLab data processing

Code for preparing and validating recordings from the instrumented bicycle.
Raw recordings and generated results are kept outside the code release.

## Modules

| Directory | Purpose |
|---|---|
| `raw_data_process/` | Crop, export, merge and validate recorded sensor streams. |
| `data_analysis/scripts/` | Dataset-validation figures and timing tables. |
| `headpose_estimation/` | Camera/helmet calibration, head pose and gaze processing. |
| `lidar2lidar_calibration/` | LiDAR calibration and camera–LiDAR projection tools. |
| `gps_fusion/` | Reproducible offline GNSS/IMU fusion on a test segment. |
| `breaksensor_calibration/` | Brake-sensor calibration and logging. |

## Main entry points

Use `--help` before processing a new recording.

```bash
# Trim exported streams to a time interval
python3 raw_data_process/script/cut_bikelab_streams.py --help

# Merge trimmed tables into one workbook
python3 raw_data_process/script/merge_bikelab_csvs_to_xlsx_v2.py --help

# Dataset validation
python3 data_analysis/scripts/gnss_imu_technical_validation.py --help
python3 data_analysis/scripts/riding_input_sensor_validation_raw.py --help
python3 data_analysis/scripts/p8_ego_motion_validation.py --help
python3 data_analysis/scripts/p8_speed_timing_closed_loop.py --help
python3 data_analysis/scripts/p1_bike_ego_gaze_close_loop.py --help

# Head-pose and gaze processing
python3 headpose_estimation/scripts/estimate_headpose_from_frames.py --help
python3 headpose_estimation/scripts/evaluate_gaze_abc_by_windows.py --help

# Time-synchronisation recording and analysis
python3 raw_data_process/script/time_sync_recording/record_host_time_sync.py --help
python3 raw_data_process/script/time_sync_recording/analyse_time_sync_log.py --help

# GNSS/IMU fusion (ROS 2)
bash gps_fusion/scripts/build_ros2_workspace.sh
bash gps_fusion/scripts/run_four_way_comparison.sh \
  gps_fusion/results/local_runs/validation_r01
bash gps_fusion/scripts/postprocess_four_way_run.sh \
  gps_fusion/results/local_runs/validation_r01 \
  gps_fusion/results/local_runs/validation_r01_analysis
```

The module directories contain the YAML files and helper scripts used by these
entry points. Keep each run in a new output directory.

## Requirements

The scripts were developed on Ubuntu 22.04 with Python 3 and ROS 2 Humble.
Install the dependencies required by the selected module. GNSS bag export uses
[`ros2_unbag`](https://github.com/ika-rwth-aachen/ros2_unbag); ROS 2 fusion also
requires `robot_localization`.

## Data safety

Processing scripts read raw data and write to a new output directory. Do not
overwrite raw recordings or previous results. This cleanup adds no raw
recordings or generated results; local bags, CSV/XLSX files, figures, logs and
ROS build/install directories should remain outside commits.
