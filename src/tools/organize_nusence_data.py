import os
import json
import numpy as np
import cv2
import copy

from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import BoxVisibility

DATA_PATH = '../../data/nuscenes/'
OUT_PATH = DATA_PATH + 'organize'
SPLITS = {
    'mini_val': 'v1.0-mini',
    'mini_train': 'v1.0-mini',
}
DEBUG = False
CATS = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle',
        'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']
SENSOR_ID = {'CAM_FRONT': 1}
USED_SENSOR = ['CAM_FRONT']
RADARS_FOR_CAMERA = {
    'CAM_FRONT':       ["RADAR_FRONT"]
}
NUM_SWEEPS = 1
suffix1 = '_{}sweeps'.format(NUM_SWEEPS) if NUM_SWEEPS > 1 else ''
OUT_PATH = OUT_PATH + suffix1 + '/'
CAT_IDS = {v: i + 1 for i, v in enumerate(CATS)}

SCENE_SPLITS = {
'mini_train':
    ['scene-0061'],
'mini_val':
    ['scene-0103', 'scene-0916'],
}


def main():
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)
    for split in SPLITS:
        data_path = DATA_PATH
        nusc = NuScenes(version=SPLITS[split], dataroot=data_path, verbose=True)
        out_path = OUT_PATH + '{}.json'.format(split)
        categories_info = [{'name': CATS[i], 'id': i + 1} for i in range(len(CATS))]
        ret = {'fileName': [], 'params': [], "annotations": [], 'categories': categories_info}

        sample_num = 0
        #ergodic all sample
        for sample in nusc.sample:
            scene_name = nusc.get('scene', sample['scene_token'])['name']
            if not (split in ['test']) and \
                    not (scene_name in SCENE_SPLITS[split]):
                continue

            for sensor_name in sample['data']:
                if sensor_name in USED_SENSOR:
                    sample_num += 1
                    image_token = sample['data'][sensor_name]
                    image_data = nusc.get('sample_data', image_token)
                    image_file_name = image_data['filename']
                    radar_sensor = RADARS_FOR_CAMERA[sensor_name]
                    sample_data_token = sample['data'][radar_sensor[0]]
                    current_sd_rec = nusc.get('sample_data', sample_data_token)
                    radar_file_name = current_sd_rec['filename']
                    width = image_data['width']
                    height = image_data['height']
                    fileName_info = {
                                     'sample_token': sample['token'],
                                     'width': width,
                                     'height': height,
                                     'imageName': image_file_name,
                                     'radarName': radar_file_name
                                     }
                    ret['fileName'].append(fileName_info)
                    sd_record = nusc.get('sample_data', image_token)
                    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
                    pose_R = pose_record['rotation']
                    pose_T = pose_record['translation']
                    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                    cam_R = cs_record['rotation']
                    cam_T = cs_record['translation']
                    cam_Intrinsic = cs_record['camera_intrinsic']
                    current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
                    pose_cur_R = current_pose_rec['rotation']
                    pose_cur_T = current_pose_rec['translation']
                    current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
                    radar_R = current_cs_rec['rotation']
                    radar_T = current_cs_rec['translation']
                    params_info = {
                        'ego_to_global_r': pose_R,
                        'ego_to_global_t': pose_T,
                        'cam_r': cam_R,
                        'cam_t': cam_T,
                        'cam_Intrinsic': cam_Intrinsic,
                        'ego_cur_r': pose_cur_R,
                        'ego_cur_t': pose_cur_T,
                        'radar_r': radar_R,
                        'radar_t': radar_T
                    }
                    ret['params'].append(params_info)
                    _, boxes, camera_intrinsic = nusc.get_sample_data(image_token, box_vis_level=BoxVisibility.ANY)
                    anns = []
                    for box in boxes:
                        center = box.center
                        wlh = box.wlh
                        name = box.name
                        rotation_matrix = box.rotation_matrix
                        vel = nusc.box_velocity(box.token).tolist()
                        ann = {
                            'name': name,
                            "center": center.tolist(),
                            "wlh": wlh.tolist(),
                            "rotation_matrix": rotation_matrix.tolist(),
                            "vel": vel
                        }
                        anns.append(ann)
                    ret['annotations'].append(anns)
                    ret['image_num'] = sample_num
        json.dump(ret, open(out_path, 'w'), indent=1)


if __name__ == '__main__':
    main()

