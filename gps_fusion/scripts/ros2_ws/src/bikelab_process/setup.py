import os
from glob import glob
from setuptools import find_packages, setup

package_name = "bikelab_process"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "FUSION_GUIDE.md"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="bikelab_ws",
    maintainer_email="zhangxinyu0428@gmail.com",
    description="ROS 2 tools for GNSS/IMU test-segment replay and fusion",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "imu_file_player = bikelab_process.imu_file_player:main",
            "velned_to_twist = bikelab_process.velned_to_twist:main",
            "gnss_course_to_imu = bikelab_process.gnss_course_to_imu:main",
            "fix_after_course_gate = bikelab_process.fix_after_course_gate:main",
            "comparison_sensor_gate = "
            "bikelab_process.comparison_sensor_gate:main",
            "trajectory_export_plot = bikelab_process.trajectory_export_plot:main",
            "trajectory_plot_svg = bikelab_process.trajectory_plot_svg:main",
            "fusion_preflight = bikelab_process.fusion_preflight:main",
            "fusion_result_evaluate = bikelab_process.fusion_result_evaluate:main",
            "four_way_compare_evaluate = "
            "bikelab_process.four_way_compare_evaluate:main",
            "rtk_quality_evaluate = "
            "bikelab_process.rtk_quality_evaluate:main",
        ],
    },
)
