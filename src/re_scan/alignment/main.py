import register
import fiducial
import numpy as np

from pathlib import Path

# Define the paths
ROOT_DIR = Path(__file__).parent.parent
DATASET_DIR_1 = ROOT_DIR / 'data_non_clipped_tsdf'
DATASET_DIR_2 = ROOT_DIR / 'data_refine/sdk'

def main():


    # _, _, _ = register.register_point_clouds_via_apriltag(
    #     root_dir=ROOT_DIR,
    #     data_dir_1=DATASET_DIR_1,
    #     data_dir_2=DATASET_DIR_2,
    #     output_dir=DATASET_DIR_1,
    #     debug=True
    # )

    intrinsics = np.array([
        [552.0291012161067,   0.0,   320.0],
        [  0.0,  552.0291012161067, 240.0],
        [  0.0,    0.0,    1.0]
    ])

    T_align, best_frame_idx_1, best_frame_idx_2 = register.register_point_clouds_via_apriltag_o3d(
        root_dir=ROOT_DIR,
        target_path=DATASET_DIR_1,
        source_path=DATASET_DIR_2,
        intrinsics=intrinsics,
        output_dir=DATASET_DIR_2,
        trans_poses=True,
        debug=True
    )


if __name__ == "__main__":
    main()