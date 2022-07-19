import json
import os
import cv2
import imageio
import numpy as np
from torch.utils.data import Dataset


def downsample(img, downsample):
    return cv2.resize(img, (0, 0), fx=downsample, fy=downsample, interpolation=cv2.INTER_AREA)


class Blender(Dataset):
    def __init__(self, data_dir, data_type, downsample):
        data, params = load_blender(data_dir, data_type)
        self.ori_images = data["images"]
        self.c2ws = data["c2ws"]
        self.params = params
        self.images = self.ori_images
        self.set_downsample(downsample)

    def __getitem__(self, item):
        return {"images": self.images[item], "c2ws": self.c2ws[item]}

    def __len__(self):
        return len(self.images)

    def shuffle(self):
        np.random.shuffle(self.images)
        np.random.shuffle(self.c2ws)

    def set_downsample(self, downsample):
        self.images = [downsample(img, downsample) for img in self.ori_images]


def load_blender(data_dir, data_type):
    with open(os.path.join(data_dir, "transforms_{}.json".format(data_type))) as jsfile:
        transforms = json.load(jsfile)
    images = []
    c2ws = []
    for frame in transforms['frames']:
        img = np.array(imageio.imread(os.path.join(data_dir, frame['file_path'] + ".png")))[..., :3] / 255.0
        images.append(img)
        c2ws.append(np.array(frame['transform_matrix']))
    height, width = images[0].shape[:2]
    camera_angle_x = float(transforms['camera_angle_x'])
    params = {
        "height": height,
        "width": width,
        "focal": 0.5 * width / np.tan(0.5 * camera_angle_x),
        "near": 2.0,
        "far": 6.0,
        "length": len(images)
    }
    images = np.array(images)
    c2ws = np.array(c2ws)
    return {"images": images, "c2ws": c2ws}, params
