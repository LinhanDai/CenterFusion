# Copyright (c) Xingyi Zhou. All Rights Reserved
'''
nuScenes pre-processing script.
This file convert the nuScenes annotation into COCO format.
'''
import os
import json
import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from nuScenes_lib.utils_kitti import KittiDB
from nuscenes.eval.detection.utils import category_to_detection_name
from pyquaternion import Quaternion

import _init_paths
from src.lib.utils.ddd_utils import compute_box_3d, project_to_image, alpha2rot_y
from src.lib.utils.ddd_utils import draw_box_3d, unproject_2d_to_3d
from src.lib.utils.pointcloud import RadarPointCloudWithVelocity as RadarPointCloud
from nuScenes_lib.utils_radar import map_pointcloud_to_image
import time

DATA_PATH = '../../data/nuscenes/'
OUT_PATH = DATA_PATH + 'annotations'
SPLITS = {
    'mini_val': 'v1.0-mini',
    'mini_train': 'v1.0-mini',
}

DEBUG = True
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


def _rot_y2alpha(rot_y, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    alpha = rot_y - np.arctan2(x - cx, fx)
    if alpha > np.pi:
        alpha -= 2 * np.pi
    if alpha < -np.pi:
        alpha += 2 * np.pi
    return alpha


def _bbox_inside(box1, box2):
    return box1[0] > box2[0] and box1[0] + box1[2] < box2[0] + box2[2] and \
           box1[1] > box2[1] and box1[1] + box1[3] < box2[1] + box2[3]


def main():
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)
    for split in SPLITS:
        # data_path = DATA_PATH + '{}/'.format(SPLITS[split])
        data_path = DATA_PATH
        nusc = NuScenes(
            version=SPLITS[split], dataroot=data_path, verbose=True)
        out_path = OUT_PATH + '{}.json'.format(split)
        categories_info = [{'name': CATS[i], 'id': i + 1} for i in range(len(CATS))]
        ret = {'images': [], 'annotations': [], 'categories': categories_info,
               'videos': []}
        num_images = 0
        num_anns = 0
        num_videos = 0
        # A "sample" in nuScenes refers to a timestamp with 6 cameras and 1 LIDAR.
        for sample in nusc.sample:
            scene_name = nusc.get('scene', sample['scene_token'])['name']
            if not (split in ['test']) and \
                    not (scene_name in SCENE_SPLITS[split]):
                continue
            if sample['prev'] == '':
                print('scene_name', scene_name)
                num_videos += 1
                ret['videos'].append({'id': num_videos, 'file_name': scene_name})
                frame_ids = {k: 0 for k in sample['data']}
            # We decompose a sample into 6 images in our case.
            for sensor_name in sample['data']:
                if sensor_name in USED_SENSOR:
                    image_token = sample['data'][sensor_name]
                    image_data = nusc.get('sample_data', image_token)
                    num_images += 1

                    # Complex coordinate transform. This will take time to understand.
                    sd_record = nusc.get('sample_data', image_token)
                    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
                    global_from_car = transform_matrix(pose_record['translation'],
                                                       Quaternion(pose_record['rotation']), inverse=False)
                    cs_record = nusc.get(
                        'calibrated_sensor', sd_record['calibrated_sensor_token'])
                    car_from_sensor = transform_matrix(
                        cs_record['translation'], Quaternion(cs_record['rotation']),
                        inverse=False)
                    trans_matrix = np.dot(global_from_car, car_from_sensor)

                    vel_global_from_car = transform_matrix(np.array([0, 0, 0]),
                                                           Quaternion(pose_record['rotation']), inverse=False)
                    vel_car_from_sensor = transform_matrix(np.array([0, 0, 0]),
                                                           Quaternion(cs_record['rotation']), inverse=False)
                    velocity_trans_matrix = np.dot(vel_global_from_car, vel_car_from_sensor)

                    _, boxes, camera_intrinsic = nusc.get_sample_data(
                        image_token, box_vis_level=BoxVisibility.ANY)
                    calib = np.eye(4, dtype=np.float32)
                    calib[:3, :3] = camera_intrinsic
                    calib = calib[:3]
                    frame_ids[sensor_name] += 1

                    # get radar pointclouds
                    all_radar_pcs = RadarPointCloud(np.zeros((18, 0)))
                    for radar_channel in RADARS_FOR_CAMERA[sensor_name]:
                        radar_pcs, _ = RadarPointCloud.from_file_multisweep(nusc,
                                                                            sample, radar_channel, sensor_name,
                                                                            NUM_SWEEPS)
                        all_radar_pcs.points = np.hstack((all_radar_pcs.points, radar_pcs.points))

                    # image information in COCO format
                    image_info = {'id': num_images,
                                  'file_name': image_data['filename'],
                                  'calib': calib.tolist(),
                                  'video_id': num_videos,
                                  'frame_id': frame_ids[sensor_name],
                                  'sensor_id': SENSOR_ID[sensor_name],
                                  'sample_token': sample['token'],
                                  'trans_matrix': trans_matrix.tolist(),
                                  'velocity_trans_matrix': velocity_trans_matrix.tolist(),
                                  'width': sd_record['width'],
                                  'height': sd_record['height'],
                                  'pose_record_trans': pose_record['translation'],
                                  'pose_record_rot': pose_record['rotation'],
                                  'cs_record_trans': cs_record['translation'],
                                  'cs_record_rot': cs_record['rotation'],
                                  'radar_pc': all_radar_pcs.points.tolist(),
                                  'camera_intrinsic': camera_intrinsic.tolist(),
                                  }
                    ret['images'].append(image_info)

                    anns = []
                    for box in boxes:
                        det_name = category_to_detection_name(box.name)
                        if det_name is None:
                            continue
                        num_anns += 1
                        v = np.dot(box.rotation_matrix, np.array([1, 0, 0]))
                        yaw = -np.arctan2(v[2], v[0])
                        box.translate(np.array([0, box.wlh[2] / 2, 0]))
                        category_id = CAT_IDS[det_name]

                        amodel_center = project_to_image(
                            np.array([box.center[0], box.center[1] - box.wlh[2] / 2, box.center[2]],
                                     np.float32).reshape(1, 3), calib)[0].tolist()

                        vel = nusc.box_velocity(box.token).tolist()  # global frame
                        # vel = np.dot(np.linalg.inv(trans_matrix),
                        #   np.array([vel[0], vel[1], vel[2], 0], np.float32)).tolist()

                        # get velocity in camera coordinates
                        vel_cam = np.dot(np.linalg.inv(velocity_trans_matrix),
                                         np.array([vel[0], vel[1], vel[2], 0], np.float32)).tolist()
                        # vel_glob = np.dot(velocity_trans_matrix,
                        #   np.array([vel_cam[0], vel_cam[1], vel_cam[2], 0], np.float32)).tolist()

                        # instance information in COCO format
                        ann = {
                            'id': num_anns,
                            'image_id': num_images,
                            'category_id': category_id,
                            'dim': [box.wlh[2], box.wlh[0], box.wlh[1]],
                            'location': [box.center[0], box.center[1], box.center[2]],
                            'depth': box.center[2],
                            'occluded': 0,
                            'truncated': 0,
                            'rotation_y': yaw,
                            'amodel_center': amodel_center,
                            'iscrowd': 0,
                            'velocity': vel,
                            'velocity_cam': vel_cam
                        }

                        bbox = KittiDB.project_kitti_box_to_image(
                            copy.deepcopy(box), camera_intrinsic, imsize=(1600, 900))
                        alpha = _rot_y2alpha(yaw, (bbox[0] + bbox[2]) / 2,
                                             camera_intrinsic[0, 2], camera_intrinsic[0, 0])
                        ann['bbox'] = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                        ann['area'] = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        ann['alpha'] = alpha
                        anns.append(ann)

                    # Filter out bounding boxes outside the image
                    visable_anns = []
                    for i in range(len(anns)):
                        vis = True
                        for j in range(len(anns)):
                            if anns[i]['depth'] - min(anns[i]['dim']) / 2 > \
                                    anns[j]['depth'] + max(anns[j]['dim']) / 2 and \
                                    _bbox_inside(anns[i]['bbox'], anns[j]['bbox']):
                                vis = False
                                break
                        if vis:
                            visable_anns.append(anns[i])
                        else:
                            pass

                    for ann in visable_anns:
                        ret['annotations'].append(ann)

                    if DEBUG:
                        img_path = data_path + image_info['file_name']
                        img = cv2.imread(img_path)
                        img_3d = img.copy()
                        # plot radar point clouds
                        pc = np.array(image_info['radar_pc'])
                        cam_intrinsic = np.array(image_info['calib'])[:, :3]
                        points, coloring, _ = map_pointcloud_to_image(pc, cam_intrinsic)
                        for i, p in enumerate(points.T):
                            img = cv2.circle(img, (int(p[0]), int(p[1])), 5, (255, 0, 0), -1)

                        for ann in visable_anns:
                            bbox = ann['bbox']
                            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                                          (int(bbox[2] + bbox[0]), int(bbox[3] + bbox[1])),
                                          (0, 0, 255), 3, lineType=cv2.LINE_AA)
                            box_3d = compute_box_3d(ann['dim'], ann['location'], ann['rotation_y'])
                            box_2d = project_to_image(box_3d, calib)
                            img_3d = draw_box_3d(img_3d, box_2d)

                            pt_3d = unproject_2d_to_3d(ann['amodel_center'], ann['depth'], calib)
                            pt_3d[1] += ann['dim'][0] / 2
                            print('location', ann['location'])
                            print('loc model', pt_3d)
                            pt_2d = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                                             dtype=np.float32)
                            pt_3d = unproject_2d_to_3d(pt_2d, ann['depth'], calib)
                            pt_3d[1] += ann['dim'][0] / 2
                            print('loc      ', pt_3d)
                        cv2.imshow('img', img)
                        cv2.imshow('img_3d', img_3d)
                        cv2.waitKey(100)

                        # cv2.imwrite('img.jpg', img)
                        # cv2.imwrite('img_3d.jpg', img_3d)
                        # nusc.render_sample_data(image_token, out_path='nusc_img.jpg')
                        #input('press enter to continue')
                        # plt.show()\

        # print('reordering images')
        # images = ret['images']
        # video_sensor_to_images = {}
        # for image_info in images:
        #     tmp_seq_id = image_info['video_id'] * 20 + image_info['sensor_id']
        #     if tmp_seq_id in video_sensor_to_images:
        #         video_sensor_to_images[tmp_seq_id].append(image_info)
        #     else:
        #         video_sensor_to_images[tmp_seq_id] = [image_info]
        # ret['images'] = []
        # for tmp_seq_id in sorted(video_sensor_to_images):
        #     ret['images'] = ret['images'] + video_sensor_to_images[tmp_seq_id]
        #
        # print('{} {} images {} boxes'.format(
        #     split, len(ret['images']), len(ret['annotations'])))
        # print('out_path', out_path)
        json.dump(ret, open(out_path, 'w'))


# Official train/ val split from
# https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/splits.py
SCENE_SPLITS = {
'mini_train':
    ['scene-0061'],
'mini_val':
    ['scene-0103', 'scene-0916'],
}
if __name__ == '__main__':
    main()
