import os
import json
import numpy as np
import cv2
import copy

import _init_paths
from src.lib.utils.ddd_utils import compute_box_3d, project_to_image, alpha2rot_y
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix, view_points
from nuScenes_lib.utils_kitti import KittiDB
from src.lib.utils.pointcloud import RadarPointCloudWithVelocity as RadarPointCloud
from nuScenes_lib.utils_radar import map_pointcloud_to_image
from src.lib.utils.ddd_utils import draw_box_3d, unproject_2d_to_3d


CATS = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle',
        'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']

DATA_PATH = '../../data/nuscenes/'
OUT_PATH = DATA_PATH + 'annotations'
NUM_SWEEPS = 1
suffix1 = '_{}sweeps'.format(NUM_SWEEPS) if NUM_SWEEPS > 1 else ''
OUT_PATH = OUT_PATH + suffix1 + '/'

INPUT_PATH = DATA_PATH + 'organize'
SPLITS = {
    'mini_val': 'v1.0-mini',
    'mini_train': 'v1.0-mini',
}
CAT_IDS = {v: i + 1 for i, v in enumerate(CATS)}

DEBUG = True


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


def get_corners(box, wlh_factor: float = 1.0) -> np.ndarray:
    """
    Returns the bounding box corners.
    :param wlh_factor: Multiply w, l, h by a factor to scale the box.
    :return: <np.float: 3, 8>. First four corners are the ones facing forward.
        The last four are the ones facing backwards.
    """
    w, l, h = np.array(box['wlh']) * wlh_factor

    # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
    x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
    z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners))

    # Rotate
    corners = np.dot(box['rotation_matrix'], corners)

    # Translate
    x, y, z = box['center']
    corners[0, :] = corners[0, :] + x
    corners[1, :] = corners[1, :] + y
    corners[2, :] = corners[2, :] + z

    return corners


def box_to_image(box, p_left, imsize):
    box['center'][1] = box['center'][1] - box['wlh'][2] / 2
    # Check that some corners are inside the image.
    corners = np.array([corner for corner in get_corners(box).T if corner[2] > 0]).T
    if len(corners) == 0:
        return None

    # Project corners that are in front of the camera to 2d to get bbox in pixel coords.
    imcorners = view_points(corners, p_left, normalize=True)[:2]
    bbox = (np.min(imcorners[0]), np.min(imcorners[1]), np.max(imcorners[0]), np.max(imcorners[1]))

    # Crop bbox to prevent it extending outside image.
    bbox_crop = tuple(max(0, b) for b in bbox)
    bbox_crop = (min(imsize[0], bbox_crop[0]),
                 min(imsize[0], bbox_crop[1]),
                 min(imsize[0], bbox_crop[2]),
                 min(imsize[1], bbox_crop[3]))

    # Detect if a cropped box is empty.
    if bbox_crop[0] >= bbox_crop[2] or bbox_crop[1] >= bbox_crop[3]:
        return None

    return bbox_crop




def main():
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)
    for split in SPLITS:
        data_path = DATA_PATH
        out_path = OUT_PATH + '/{}.json'.format(split)
        input_path = INPUT_PATH + '/{}.json'.format(split)
        input_json = json.load(open(input_path, 'r'))
        sample_num = input_json['image_num']
        dataroot = '../../data/nuscenes/'
        file_info = input_json['fileName']
        params_info = input_json['params']
        annotations_info = input_json['annotations']
        categories_info = input_json['categories']
        ret = {'images': [], 'annotations': [], 'categories': categories_info,
               'videos': []}

        num_images = 0
        num_anns = 0
        for i in range(sample_num):
            num_images += 1
            pose_R = params_info[i]['ego_to_global_r']
            pose_T = params_info[i]['ego_to_global_t']
            global_from_car = transform_matrix(pose_T, Quaternion(pose_R), inverse=False)
            cam_R = params_info[i]['cam_r']
            cam_T = params_info[i]['cam_t']
            car_from_sensor = transform_matrix(cam_T, Quaternion(cam_R), inverse=False)
            trans_matrix = np.dot(global_from_car, car_from_sensor)
            vel_global_from_car = transform_matrix(np.array([0, 0, 0]), Quaternion(pose_R), inverse=False)
            vel_car_from_sensor = transform_matrix(np.array([0, 0, 0]), Quaternion(cam_R), inverse=False)
            velocity_trans_matrix = np.dot(vel_global_from_car, vel_car_from_sensor)
            calib = np.eye(4, dtype=np.float32)
            calib[:3, :3] = params_info[i]['cam_Intrinsic']
            calib = calib[:3]
            all_radar_pcs = RadarPointCloud(np.zeros((18, 0)))
            radar_pcs = RadarPointCloud.read_radar_data(dataroot, file_info[i], params_info[i])
            all_radar_pcs.points = np.hstack((all_radar_pcs.points, radar_pcs.points))
            # image information in COCO format
            image_info = {'id': num_images,
                          'file_name': file_info[i]['imageName'],
                          'calib': calib.tolist(),
                          'video_id': 1,
                          'frame_id': num_images,
                          'sensor_id': 1,
                          'sample_token': file_info[i]['sample_token'],
                          'trans_matrix': trans_matrix.tolist(),
                          'velocity_trans_matrix': velocity_trans_matrix.tolist(),
                          'width': file_info[i]['width'],
                          'height': file_info[i]['height'],
                          'pose_record_trans': params_info[i]['ego_to_global_t'],
                          'pose_record_rot': params_info[i]['ego_to_global_r'],
                          'cs_record_trans': params_info[i]['cam_t'],
                          'cs_record_rot': params_info[i]['cam_r'],
                          'radar_pc': all_radar_pcs.points.tolist(),
                          'camera_intrinsic': params_info[i]['cam_Intrinsic'],
                          }
            ret['images'].append(image_info)
            boxes = annotations_info[i]
            anns = []
            for box in boxes:
                det_name = category_to_detection_name(box['name'])
                if det_name is None:
                    continue
                num_anns += 1
                v = np.dot(box['rotation_matrix'], np.array([1, 0, 0]))
                yaw = -np.arctan2(v[2], v[0])
                box['center'][1] = box['center'][1] + box['wlh'][2] / 2
                category_id = CAT_IDS[det_name]
                amodel_center = project_to_image(
                    np.array([box['center'][0], box['center'][1] - box['wlh'][2] / 2, box['center'][2]],
                             np.float32).reshape(1, 3), calib)[0].tolist()
                vel = box['vel']
                # get velocity in camera coordinates
                vel_cam = np.dot(np.linalg.inv(velocity_trans_matrix),
                                 np.array([vel[0], vel[1], vel[2], 0], np.float32)).tolist()

                ann = {
                    'id': num_anns,
                    'image_id': num_images,
                    'category_id': category_id,
                    'dim': [box['wlh'][2], box['wlh'][0], box['wlh'][1]],
                    'location': [box['center'][0], box['center'][1], box['center'][2]],
                    'depth': box['center'][2],
                    'occluded': 0,
                    'truncated': 0,
                    'rotation_y': yaw,
                    'amodel_center': amodel_center,
                    'iscrowd': 0,
                    'velocity': vel,
                    'velocity_cam': vel_cam
                }
                bbox = box_to_image(copy.deepcopy(box), np.array(params_info[i]['cam_Intrinsic']), imsize=(1600, 900))
                alpha = _rot_y2alpha(yaw, (bbox[0] + bbox[2]) / 2,
                                     np.array(params_info[i]['cam_Intrinsic'])[0, 2], np.array(params_info[i]['cam_Intrinsic'])[0, 0])
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
                cv2.imshow('img', img)
                cv2.imshow('img_3d', img_3d)
                cv2.waitKey(200)
                # cv2.imwrite('img.jpg', img)
                # cv2.imwrite('img_3d.jpg', img_3d)
                # input('press enter to continue')
        json.dump(ret, open(out_path, 'w'))


if __name__ == '__main__':
    main()
