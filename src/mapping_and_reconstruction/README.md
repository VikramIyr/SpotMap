# Open3D Setup
Clone the [Open3D GitHub repository](https://github.com/isl-org/Open3D) into the main project folder, `SpotMap`.

# SLAM Parameter Configuration
Edit the SLAM parameters in `configs/spot.json` as needed.

# Module Execution
Create and activate the Conda environment:
```bash
conda env create -f environment.yml
conda activate open3d_gpu
```

Run the module using the provided script:
```bash
chmod +x run.sh
bash run.sh
```