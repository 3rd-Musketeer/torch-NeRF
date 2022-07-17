import json
import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
import cv2


def load_blender(data_dir, data_type, downsample = None):
    with open(os.path.join(data_dir, "transforms_{}.json".format(data_type))) as jsfile:
        transforms = json.load(jsfile)
    images = []
    c2ws = []
    for frame in transforms['frames']:
        img = np.array(imageio.imread(os.path.join(data_dir, frame['file_path'] + ".png")))[..., :3] / 255.0
        if downsample is not None:
            img = cv2.resize(img, (0, 0), fx=downsample, fy=downsample, interpolation=cv2.INTER_AREA)
        images.append(img)
        c2ws.append(np.array(frame['transform_matrix']))
    height, width = images[0].shape[:2]
    camera_angle_x = float(transforms['camera_angle_x'])
    params = {
        "height": height,
        "width": width,
        "focal": 0.5 * width / np.tan(0.5 * camera_angle_x),
        "near": 2.0,
        "far": 6.0
    }
    images = np.array(images)
    c2ws = np.array(c2ws)
    return {"images": images, "c2ws": c2ws, "length": len(images)}, params
