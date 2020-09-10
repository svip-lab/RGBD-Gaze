import os

from PIL import Image
import numpy as np
import pandas as pd
import cv2

import torch as th
from torch.utils import data
from torchvision import transforms as tf

from time import time
import pickle


class GazePointAllDataset(data.Dataset):
    def __init__(self, root_dir, w_screen=59.77, h_screen=33.62, transform=None, phase="train", **kwargs):
        self.root_dir = root_dir
        self.w_screen = w_screen
        self.h_screen = h_screen
        self.transform = transform
        self.kwargs = kwargs
        self.anno = pd.read_csv(os.path.join(root_dir, phase + "_meta.csv"), index_col=0)
        # if os.path.isfile(os.path.join(root_dir, "depth_stat.pkl")):
        #     with open(os.path.join(root_dir, "depth_stat.pkl"), "rb") as fp:
        #         stat = pickle.load(fp)
        #         anno.drop(anno.iloc[stat[0]])
        root_dir = root_dir.rstrip("/").rstrip("\\")
        self.face_image_list = (root_dir + "/" + self.anno["face_image"]).tolist()
        self.face_depth_list = (root_dir + "/" + self.anno["face_depth"]).tolist()
        self.face_bbox_list = (root_dir + "/" + self.anno["face_bbox"]).tolist()
        self.le_image_list = (root_dir + "/" + self.anno["left_eye_image"]).tolist()
        self.re_image_list = (root_dir + "/" + self.anno["right_eye_image"]).tolist()
        self.le_depth_list = (root_dir + "/" + self.anno["left_eye_depth"]).tolist()
        self.re_depth_list = (root_dir + "/" + self.anno["right_eye_depth"]).tolist()
        self.le_bbox_list = (root_dir + "/" + self.anno["left_eye_bbox"]).tolist()
        self.re_bbox_list = (root_dir + "/" + self.anno["right_eye_bbox"]).tolist()
        # self.le_coord_list = (root_dir + "/" + self.anno["left_eye_coord"]).tolist()
        # self.re_coord_list = (root_dir + "/" + self.anno["right_eye_coord"]).tolist()
        self.gt_name_list = (root_dir + "/" + self.anno["gaze_point"]).tolist()

        for data_item in kwargs.keys():
            if data_item not in ("face_image", "face_depth", "eye_image", "eye_depth",
                                 "face_bbox", "eye_bbox", "gt", "eye_coord", "info"):
                raise ValueError(f"unrecognized dataset item: {data_item}")

    def __len__(self):
        return len(self.face_image_list)

    def __getitem__(self, idx):
        with open(self.le_bbox_list[idx]) as fp:
            le_bbox = list(map(float, fp.readline().split()))
        with open(self.re_bbox_list[idx]) as fp:
            re_bbox = list(map(float, fp.readline().split()))
        with open(self.face_bbox_list[idx]) as fp:
            face_bbox = list(map(float, fp.readline().split()))

        le_coor = np.load(self.le_coord_list[idx])
        re_coor = np.load(self.re_coord_list[idx])
        gt = np.load(self.gt_name_list[idx])

        gt[0] -= self.w_screen / 2
        gt[1] -= self.h_screen / 2

        sample = {}

        sample["index"] = th.LongTensor([idx])
        index = f"{self.anno.index[idx]:010d}"
        sample["pid"] = th.LongTensor([int(index[:5])])
        sample["sid"] = th.LongTensor([int(index[5:])])

        sample['gt'] = th.FloatTensor(gt)

        if self.kwargs.get('face_image'):
            face_image = Image.open(self.face_image_list[idx])
            sample['face_image'] = self.transform(face_image) if self.transform is not None else face_image

        if self.kwargs.get('face_depth'):
            assert np.abs((face_bbox[3] - face_bbox[1]) - (face_bbox[2] - face_bbox[0])) <= 2, f"invalid face bbox @ {self.face_bbox_list[idx]}"
            scale_factor = 224 / (face_bbox[2] - face_bbox[0])
            # scale_factor = min(scale_factor, 1.004484)
            # scale_factor = max(scale_factor, 0.581818)
            face_depth = cv2.imread(self.face_depth_list[idx], -1)
            # face_depth = np.int32(face_depth)
            # face_depth[face_depth<500] = 500
            # face_depth[face_depth > 1023] = 1023
            # face_depth -= 512
            if self.transform is not None:
                face_depth = face_depth[np.newaxis, :, :]# / scale_factor
                # sample['face_depth'] = th.FloatTensor(face_depth / 883)
                sample['face_depth'] = th.clamp((th.FloatTensor(face_depth.astype('float')) - 500) / 500, 0., 1.)
                sample['face_scale_factor'] = th.FloatTensor([scale_factor])
            else:
                sample['face_depth'] = face_depth
            # print('max: {}, min:{}'.format((face_depth / 430).max(), (face_depth / 430).min()), flush=True)

        if self.kwargs.get('eye_image'):
            le_image = Image.open(self.le_image_list[idx])
            re_image = Image.open(self.re_image_list[idx])
            sample['left_eye_image'] = self.transform(le_image) if self.transform is not None else le_image
            sample['right_eye_image'] = self.transform(re_image) if self.transform is not None else re_image

        if self.kwargs.get('eye_depth'):
            le_depth = cv2.imread(self.le_depth_list[idx], -1)
            re_depth = cv2.imread(self.re_depth_list[idx], -1)
            if self.transform is not None:
                le_depth = le_depth[np.newaxis, :, :].astype('float') # / le_scale_factor  # the new dim is the dim with np.newaxis
                re_depth = re_depth[np.newaxis, :, :].astype('float') # / re_scale_factor
                # sample['left_depth'] = torch.FloatTensor(le_depth/1000)
                # sample['right_depth'] = torch.FloatTensor(re_depth/1000)
                sample['left_eye_depth'] = th.FloatTensor(le_depth)
                sample['right_eye_depth'] = th.FloatTensor(re_depth)
            else:
                sample['left_eye_depth'] = le_depth
                sample['right_eye_depth'] = re_depth

        if self.kwargs.get('eye_bbox'):
            assert le_bbox[3] - le_bbox[1] == le_bbox[2] - le_bbox[0], f"invalid left eye bbox @ {self.le_bbox_list[idx]}"
            le_scale_factor = 224 / (le_bbox[2] - le_bbox[0])
            # le_scale_factor = min(le_scale_factor, 1.004484)
            # le_scale_factor = max(le_scale_factor, 0.581818)
            assert re_bbox[3] - re_bbox[1] == re_bbox[2] - re_bbox[0], f"invalid right eye bbox @ {self.re_bbox_list[idx]}"
            re_scale_factor = 224 / (re_bbox[2] - re_bbox[0])
            # re_scale_factor = min(re_scale_factor, 1.004484)
            # re_scale_factor = max(re_scale_factor, 0.581818)
            sample["left_eye_scale_factor"] = th.FloatTensor([le_scale_factor])
            sample["right_eye_scale_factor"] = th.FloatTensor([re_scale_factor])
            sample['left_eye_bbox'] = th.FloatTensor(le_bbox)
            sample['right_eye_bbox'] = th.FloatTensor(re_bbox)

        if self.kwargs.get('face_bbox'):
            sample['face_bbox'] = th.FloatTensor(face_bbox)

        if self.kwargs.get('eye_coord'):
            sample['left_eye_coord'] = th.FloatTensor(np.float32(le_coor))
            sample['right_eye_coord'] = th.FloatTensor(np.float32(re_coor))

        if self.kwargs.get('info'):
            le_depth = np.clip((cv2.imread(self.le_depth_list[idx], -1) - 500) / 500, 0, 1.)
            re_depth = np.clip((cv2.imread(self.le_depth_list[idx], -1) - 500) / 500, 0, 1.)
            # get info
            le_depth_ = le_depth[le_depth > 0]
            if len(le_depth_) > 0:
                le_info = [le_coor[0] / 1920, le_coor[1] / 1080, np.mean(le_depth_)]
            else:
                le_info = [le_coor[0] / 1920, le_coor[1] / 1080] + [0.]

            re_depth_ = re_depth[re_depth > 0]
            if len(re_depth_) > 0:
                re_info = [re_coor[0] / 1920, re_coor[1] / 1080, np.mean(re_depth_)]
            else:
                re_info = [re_coor[0] / 1920, re_coor[1] / 1080] + [0.]
            sample['left_eye_info'] = th.FloatTensor(le_info)
            sample['right_eye_info'] = th.FloatTensor(re_info)

        return sample


if __name__ == '__main__':
    from tqdm import tqdm
    dataset = GazePointAllDataset(
        root_dir=r"D:\\data\\gaze",
        phase="train",
        face_image=True,
        face_depth=True,
        face_bbox=True,
        eye_image=True,
        eye_depth=True,
        eye_bbox=True,
        eye_coord=True
    )

    for sample in tqdm(dataset, desc="testing"):
        pass
