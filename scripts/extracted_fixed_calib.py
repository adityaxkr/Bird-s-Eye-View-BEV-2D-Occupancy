# scripts/extract_fixed_calib.py
from data.nuscenes_loader import BEVOccupancyDataset
import config.config as cfg
from data.preprocess import preprocess_intrinsic, preprocess_extrinsic
import numpy as np

ds = BEVOccupancyDataset(dataroot=cfg.DATAROOT, version=cfg.VERSION)
nusc = ds.nusc

scene = nusc.scene[0]
sample = nusc.get("sample", scene["first_sample_token"])

K_list, E_list = [], []
for cam_name in cfg.CAM_NAMES:
    cam_token = sample["data"][cam_name]
    cam_data  = nusc.get("sample_data", cam_token)
    calib     = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])

    K = preprocess_intrinsic(
        calib["camera_intrinsic"],
        orig_w=cam_data["width"],
        orig_h=cam_data["height"]
    ).numpy()
    E = preprocess_extrinsic(
        calib["rotation"],
        calib["translation"]
    ).numpy()

    K_list.append(K)
    E_list.append(E)

np.save("fixed_K.npy", np.stack(K_list))  # (6,3,3)
np.save("fixed_E.npy", np.stack(E_list))  # (6,4,4)
print("saved fixed_K.npy & fixed_E.npy")