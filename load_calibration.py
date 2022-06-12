import os, sys
sys.path.insert(0,os.getcwd())
import numpy as np
from tridet.structures.pose import Pose
import cv2
import pandas as pd

def _read_calibration_file(filename):
        """Reads a calibration file and creates corresponding pose, camera calibration objects.
        Reference frame (world frame) is camera 0.

        Returns
        -------
        calibration_table: dict, default: None
            Calibration table used for looking up sample-level calibration.
            >>> (K, pose_0S) = self.calibration_table[(calibration_key, datum_name)]

            Calibration key is just filename prefix. (i.e. 007431)
        """
        index = os.path.basename(filename).split(".")[0]
        calibration = pd.read_csv(filename, delim_whitespace=True, header=None)

        # Camera-to-camera rectification
        R0_rect = np.eye(4)
        R0_rect[:3, :3] = calibration.loc[4].values[1:10].reshape(-1, 3).astype(np.float64) #3-by-3 camera

        # P_20 is projection matrix from camera 0 to 2 (forward facing RGB camera - 3rd line KITTI calib CSV)
        # >>> X_2 = P_20 * R0_rect * X_0
        # See here for details https://github.com/bostondiditeam/kitti/blob/master/Papers_Summary/Geiger2013IJRR/readme.md
        P_20 = calibration.loc[2].values[1:].reshape(-1, 4).astype(np.float64)
        K2, R_20, t_20 = cv2.decomposeProjectionMatrix(P_20)[:3]

        # Extrinsic transformation from camera 0 (world frame) to camera 2
        T_20 = np.eye(4)
        T_20[:3, :3] = R_20[:3, :3]
        T_20[:3, 3] = (t_20[:3] / t_20[3]).squeeze()

        # Rectified pose for camera 0 to 2
        pose_20 = Pose.from_matrix(T_20 @ R0_rect)

        # Right camera (camera 3, `image_3`) for stereo.
        P_30 = calibration.loc[3].values[1:].reshape(-1, 4).astype(np.float64)
        K3, R_30, t_30 = cv2.decomposeProjectionMatrix(P_30)[:3]

        # Extrinsic transformation from camera 0 (world frame) to camera 3
        T_30 = np.eye(4)
        T_30[:3, :3] = R_30[:3, :3]
        T_30[:3, 3] = (t_30[:3] / t_30[3]).squeeze()

        # Rectified pose for camera 0 to 3
        pose_30 = Pose.from_matrix(T_30 @ R0_rect)

        # Extrinsic transformation from velodyne to camera 0 frame (6th line of KITTI calib CVS [Tr_velo_to_cam])
        T_0V = calibration.loc[5].values[1:].reshape(-1, 4).astype(np.float64)
        T_0V = np.vstack([T_0V, np.array([0, 0, 0, 1])])

        # Lidar pose to camera 0 [Tr_velo_to_cam].
        # To project a point from LIDAR into camera, the following are equivalent:
        # >>> X_2 = P_20 * R0_rect * T_0V * X_V
        # >>> X_2 = Camera(K, p_cw=pose_20).project(T_0V * X_V)
        pose_0V = Pose.from_matrix(T_0V)
        return ((index, "camera_2"), (K2, pose_20.inverse())), \
                    ((index, "camera_3"), (K3, pose_30.inverse())), \
                    ((index, "velodyne"), (None, pose_0V))


if __name__ == '__main__':
    fname = '000000.txt'
    path = os.path.join(os.getcwd(), 'calibration', fname)
    ret = _read_calibration_file(path)
    print(ret)