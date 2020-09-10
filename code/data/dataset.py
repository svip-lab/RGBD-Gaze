from torch.utils.data import Dataset
import data.preprocessing as pre
import os
from skimage import io
import numpy as np
import torch
import cv2
import scipy.io as sio
from PIL import Image
import pandas as pd
from random import Random
import fire


d = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 1, 9: 0, 10: 1, 11: 1, 12: 1,
     13: 0, 14: 1, 15: 0, 16: 1, 17: 1, 18: 1, 19: 0, 20: 1, 21: 0, 22: 1, 23: 1, 24: 0,
     25: 1, 26: 1, 27: 1, 28: 1, 29: 1, 30: 0, 31: 1, 32: 1, 33: 0, 34: 1, 35: 1, 36: 1,
     37: 1, 38: 1, 39: 1, 40: 1, 41: 1, 42: 1, 43: 1, 44: 1, 45: 1, 46: 1, 47: 1, 48: 0,
     49: 1, 50: 1, 51: 0, 52: 1, 53: 1, 54: 1, 55: 1, 56: 1, 57: 1, 58: 1, 59: 1}

class GazeImageDataset(Dataset):
    def __init__(self, txt_file, root_dir, w_screen=59.77, h_screen=33.62, transform=None):
        self.root_dir = root_dir
        self.list = pre.imagePathTxt2List(os.path.join(self.root_dir, txt_file))
        self.w_screen = w_screen
        self.h_screen = h_screen
        self.transform = transform

        self.img_face_name_list = pre.imagePathTxt2List(os.path.join(self.root_dir, 'face_path.txt'))
        self.img_face_depth_name_list = pre.imagePathTxt2List(os.path.join(self.root_dir, 'face_path_depth.txt'))
        self.gt_name_list = pre.imagePathTxt2List(os.path.join(self.root_dir, 'xy_path.txt'))

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        img_face_name = self.img_face_name_list[idx]
        img_face_depth_name = self.img_face_depth_name_list[idx]
        gt_name = self.gt_name_list[idx]

        #le = io.imread(img_le_name)
        #re = io.imread(img_re_name)
        face = io.imread(img_face_name)
        depth = cv2.imread(img_face_depth_name, -1)
        gt = np.load(gt_name)
        gt[0] -= self.w_screen / 2
        gt[1] -= self.h_screen / 2
        # gt[0] /= W_screen
        # gt[1] /= H_screen
        sample = {'face': face, 'depth': depth, 'gt': gt}

        if self.transform:
            #sample['le'] = self.transform(le)
            #sample['re'] = self.transform(re)
            sample['face'] = self.transform(face)
            depth = depth[np.newaxis,:,:].astype('float') / 1000  # the new dim is the dim with np.newaxis
            #sample['fg'] = self.transform(fg)
            sample['depth'] = torch.FloatTensor(depth)
            #print(type(sample['fg']))
            #print(sample['fg'].size())
            #print(type(sample['fg']))
            sample['gt'] = torch.FloatTensor(gt)
            #print(type(sample['gt']))
        #print(sample)
        return sample





class GazePointDataset(Dataset):
    def __init__(self, txt_file, root_dir, w_screen=59.77, h_screen=33.62, transform=None):
        self.root_dir = root_dir
        self.list = pre.imagePathTxt2List(os.path.join(self.root_dir, txt_file))
        self.w_screen = w_screen
        self.h_screen = h_screen
        self.transform = transform

        self.img_face_name_list = pre.imagePathTxt2List(os.path.join(self.root_dir, 'face_path.txt'))
        self.img_face_depth_name_list = pre.imagePathTxt2List(os.path.join(self.root_dir, 'face_path_depth.txt'))
        self.face_bbx_name_list = [list(map(float, line.split())) for line in open(os.path.join(self.root_dir, 'bbxlist.txt'))]
        self.gt_name_list = pre.imagePathTxt2List(os.path.join(self.root_dir, 'xy_path.txt'))

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        img_face_name = self.img_face_name_list[idx]
        img_face_depth_name = self.img_face_depth_name_list[idx]
        face_bbx_name = self.face_bbx_name_list[idx]
        gt_name = self.gt_name_list[idx]

        #le = io.imread(img_le_name)
        #re = io.imread(img_re_name)
        face = io.imread(img_face_name)
        depth = cv2.imread(img_face_depth_name, -1)
        gt = np.load(gt_name)
        gt[0] -= self.w_screen / 2
        gt[1] -= self.h_screen / 2
        # gt[0] /= W_screen
        # gt[1] /= H_screen
        depth_ = depth[depth > 0]
        if len(depth_) > 0:
            info = [face_bbx_name[0]/1920, face_bbx_name[1]/1080, face_bbx_name[2]/1920, face_bbx_name[3]/1080] + [depth_.mean()/1000.]
        else:
            info = [face_bbx_name[0] / 1920, face_bbx_name[1] / 1080, face_bbx_name[2] / 1920,
                    face_bbx_name[3] / 1080] + [0.]

        sample = {'face': face, 'depth': depth, 'info': info, 'gt': gt}

        if self.transform:
            #sample['le'] = self.transform(le)
            #sample['re'] = self.transform(re)
            sample['face'] = self.transform(face)
            depth = depth[np.newaxis,:,:].astype('float') / 1000  # the new dim is the dim with np.newaxis
            #sample['fg'] = self.transform(fg)
            sample['depth'] = torch.FloatTensor(depth)
            #print(type(sample['fg']))
            #print(sample['fg'].size())
            #print(type(sample['fg']))
            sample['gt'] = torch.FloatTensor(gt)
            sample['info'] = torch.FloatTensor(info)
            #print(type(sample['gt']))
        #print(sample)
        return sample

class GazePointDatasetDebug(Dataset):
    def __init__(self, txt_file, root_dir, w_screen=59.77, h_screen=33.62, transform=None):
        self.root_dir = root_dir
        self.list = pre.imagePathTxt2List(os.path.join(self.root_dir, txt_file))
        self.w_screen = w_screen
        self.h_screen = h_screen
        self.transform = transform

        self.flag_glass_list = [d[int(line[34:39])] for line in open(os.path.join(self.root_dir, 'face_path.txt'))]
        self.img_face_name_list = pre.imagePathTxt2List(os.path.join(self.root_dir, 'face_path.txt'))
        self.img_face_depth_name_list = pre.imagePathTxt2List(os.path.join(self.root_dir, 'face_path_depth.txt'))
        self.face_bbx_name_list = [list(map(float, line.split())) for line in open(os.path.join(self.root_dir, 'bbxlist.txt'))]
        self.gt_name_list = pre.imagePathTxt2List(os.path.join(self.root_dir, 'xy_path.txt'))

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        img_face_name = self.img_face_name_list[idx]
        img_face_depth_name = self.img_face_depth_name_list[idx]
        face_bbx_name = self.face_bbx_name_list[idx]
        gt_name = self.gt_name_list[idx]
        flag = self.flag_glass_list[idx]

        #le = io.imread(img_le_name)
        #re = io.imread(img_re_name)
        face = io.imread(img_face_name)
        depth = cv2.imread(img_face_depth_name, -1)
        gt = np.load(gt_name)
        gt[0] -= self.w_screen / 2
        gt[1] -= self.h_screen / 2
        # gt[0] /= W_screen
        # gt[1] /= H_screen
        depth_ = depth[depth > 0]
        if len(depth_) > 0:
            info = [face_bbx_name[0]/1920, face_bbx_name[1]/1080, face_bbx_name[2]/1920, face_bbx_name[3]/1080] + [depth_.mean()/1000.]
        else:
            info = [face_bbx_name[0] / 1920, face_bbx_name[1] / 1080, face_bbx_name[2] / 1920,
                    face_bbx_name[3] / 1080] + [0.]

        sample = {'face': face, 'depth': depth, 'info': info, 'gt': gt, 'flag': flag}

        if self.transform:
            #sample['le'] = self.transform(le)
            #sample['re'] = self.transform(re)
            sample['face'] = self.transform(face)
            depth = depth[np.newaxis,:,:].astype('float') / 1000  # the new dim is the dim with np.newaxis
            #sample['fg'] = self.transform(fg)
            sample['depth'] = torch.FloatTensor(depth)
            #print(type(sample['fg']))
            #print(sample['fg'].size())
            #print(type(sample['fg']))
            sample['gt'] = torch.FloatTensor(gt)
            sample['info'] = torch.FloatTensor(info)
            #print(type(sample['gt']))
        #print(sample)
        return sample

class GazePointLRDataset(Dataset):
    def __init__(self, txt_file, root_dir, w_screen=59.77, h_screen=33.62, transform=None):
        self.root_dir = root_dir
        self.list = pre.imagePathTxt2List(os.path.join(self.root_dir, txt_file))
        self.w_screen = w_screen
        self.h_screen = h_screen
        self.transform = transform

        self.img_lefteye_name_list = pre.imagePathTxt2List(os.path.join(self.root_dir, 'lefteye_path.txt'))
        self.img_righteye_name_list = pre.imagePathTxt2List(os.path.join(self.root_dir, 'righteye_path.txt'))
        self.img_lefteye_depth_name_list = pre.imagePathTxt2List(os.path.join(self.root_dir, 'lefteye_path_depth.txt'))
        self.img_righteye_depth_name_list = pre.imagePathTxt2List(os.path.join(self.root_dir, 'righteye_path_depth.txt'))
        self.lefteye_coordinate_name_list = pre.imagePathTxt2List(os.path.join(self.root_dir, 'lefteye_coordinate_path.txt'))
        self.righteye_coordinate_name_list = pre.imagePathTxt2List(os.path.join(self.root_dir, 'righteye_coordinate_path.txt'))
        self.gt_name_list = pre.imagePathTxt2List(os.path.join(self.root_dir, 'xy_path.txt'))

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        img_lefteye_name = self.img_lefteye_name_list[idx]
        img_righteye_name = self.img_righteye_name_list[idx]
        img_lefteye_depth_name = self.img_lefteye_depth_name_list[idx]
        img_righteye_depth_name = self.img_lefteye_depth_name_list[idx]
        lefteye_coordinate_name = self.lefteye_coordinate_name_list[idx]
        righteye_coordinate_name = self.righteye_coordinate_name_list[idx]
        gt_name = self.gt_name_list[idx]

        # le = io.imread(img_le_name)
        # re = io.imread(img_re_name)
        le= io.imread(img_lefteye_name)
        re = io.imread(img_righteye_name)
        le_depth = cv2.imread(img_lefteye_depth_name, -1)
        re_depth = cv2.imread(img_righteye_depth_name, -1)
        le_coor = np.load(lefteye_coordinate_name)
        re_coor = np.load(righteye_coordinate_name)
        gt = np.load(gt_name)
        gt[0] -= self.w_screen / 2
        gt[1] -= self.h_screen / 2
        # gt[0] /= W_screen
        # gt[1] /= H_screen
        le_depth_ = le_depth[le_depth > 0]
        if len(le_depth_) > 0:
            le_info = [le_coor[0] / 1920, le_coor[1] / 1080, le_depth_.mean() / 1000.]
        else:
            le_info = [le_coor[0] / 1920, le_coor[1] / 1080] + [0.]

        re_depth_ = re_depth[re_depth > 0]
        if len(re_depth_) > 0:
            re_info = [re_coor[0] / 1920, re_coor[1] / 1080, re_depth_.mean() / 1000.]
        else:
            re_info = [re_coor[0] / 1920, re_coor[1] / 1080] + [0.]


        sample = {'left_eye': le, 'right_eye': re, 'left_depth': le_depth, 'right_depth': re_depth, 'gt': gt, 'left_info': le_info, 'right_info': re_info,}


        if self.transform:
            # sample['le'] = self.transform(le)
            # sample['re'] = self.transform(re)
            sample['left_eye'] = self.transform(le)
            sample['right_eye'] = self.transform(re)
            le_depth = le_depth[np.newaxis, :, :].astype('float') / 1000  # the new dim is the dim with np.newaxis
            re_depth = re_depth[np.newaxis, :, :].astype('float') / 1000
            # sample['fg'] = self.transform(fg)
            sample['left_depth'] = torch.FloatTensor(le_depth)
            sample['right_depth'] = torch.FloatTensor(re_depth)
            # print(type(sample['fg']))
            # print(sample['fg'].size())
            # print(type(sample['fg']))
            sample['gt'] = torch.FloatTensor(gt)
            sample['left_info'] = torch.FloatTensor(le_info)
            sample['right_info'] = torch.FloatTensor(re_info)
            # print(type(sample['gt']))
        # print(sample)
        return sample


class GazePointAllDataset(Dataset):
    def __init__(self, txt_file, root_dir, w_screen=59.77, h_screen=33.62, transform=None, **kwargs):
        self.root_dir = root_dir
        self.list = pre.imagePathTxt2List(os.path.join(self.root_dir, txt_file))
        self.w_screen = w_screen
        self.h_screen = h_screen
        self.transform = transform
        self.kwargs = kwargs

        self.img_face_name_list = pre.imagePathTxt2List(os.path.join(self.root_dir, 'face_path.txt'))
        self.img_face_depth_name_list = pre.imagePathTxt2List(os.path.join(self.root_dir, 'face_path_depth.txt'))
        self.img_lefteye_name_list = pre.imagePathTxt2List(os.path.join(self.root_dir, 'lefteye_path.txt'))
        self.img_righteye_name_list = pre.imagePathTxt2List(os.path.join(self.root_dir, 'righteye_path.txt'))
        self.img_lefteye_depth_name_list = pre.imagePathTxt2List(os.path.join(self.root_dir, 'lefteye_path_depth.txt'))
        self.img_righteye_depth_name_list = pre.imagePathTxt2List(
            os.path.join(self.root_dir, 'righteye_path_depth.txt'))

        self.le_bbx_name_list = [list(map(int, map(float, line.split()))) for line in
                                   open(os.path.join(self.root_dir, 'lefteye_bbxlist.txt'))]
        self.re_bbx_name_list = [list(map(int, map(float, line.split()))) for line in
                                   open(os.path.join(self.root_dir, 'righteye_bbxlist.txt'))]
        self.face_bbx_name_list = [list(map(int, map(float, line.split()))) for line in
                                   open(os.path.join(self.root_dir, 'bbxlist.txt'))]

        self.lefteye_coordinate_name_list = pre.imagePathTxt2List(
            os.path.join(self.root_dir, 'lefteye_coordinate_path.txt'))
        self.righteye_coordinate_name_list = pre.imagePathTxt2List(
            os.path.join(self.root_dir, 'righteye_coordinate_path.txt'))
        self.gt_name_list = pre.imagePathTxt2List(os.path.join(self.root_dir, 'xy_path.txt'))

        # new dataset section
        self.face_grid_name_list = pre.imagePathTxt2List(os.path.join(self.root_dir, 'face_grid_path.txt'))
        self.gaze_direction_sphere_list = pre.imagePathTxt2List(os.path.join(self.root_dir, 'gaze_direction_normal.txt'))

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        le_bbx = self.le_bbx_name_list[idx]
        re_bbx = self.re_bbx_name_list[idx]
        face_bbx = self.face_bbx_name_list[idx]
        lefteye_coordinate_name = self.lefteye_coordinate_name_list[idx]
        righteye_coordinate_name = self.righteye_coordinate_name_list[idx]
        gt_name = self.gt_name_list[idx]

        le_coor = np.load(lefteye_coordinate_name)
        re_coor = np.load(righteye_coordinate_name)
        gt = np.load(gt_name)
        gt[0] -= self.w_screen / 2
        gt[1] -= self.h_screen / 2


        sample = {}

        if self.transform:
            if self.kwargs.get('face'):
                img_face_name = self.img_face_name_list[idx]
                face = io.imread(img_face_name)
                sample['face'] = self.transform(face)
            if self.kwargs.get('eye'):
                img_lefteye_name = self.img_lefteye_name_list[idx]
                img_righteye_name = self.img_righteye_name_list[idx]
                le = io.imread(img_lefteye_name)
                re = io.imread(img_righteye_name)
                sample['left_eye'] = self.transform(le)
                sample['right_eye'] = self.transform(re)

        if self.kwargs.get('head_pose'):
            gaze_direction_sphere_name = self.gaze_direction_sphere_list[idx]
            gaze_direction = sio.loadmat(gaze_direction_sphere_name)[
                'gaze_direction_normal']
            sample['head_pose'] = torch.FloatTensor(gaze_direction.astype('float'))

        if self.kwargs.get('grid'):
            # new dataset section
            face_grid_name = self.face_grid_name_list[idx]
            face_grid = np.load(face_grid_name)
            sample['grid'] = torch.FloatTensor(face_grid.astype('float'))

        if self.kwargs.get('face_depth'):
            img_face_depth_name = self.img_face_depth_name_list[idx]
            scale_factor = 224 / (face_bbx[2] - face_bbx[0])
            scale_factor = min(scale_factor, 1.004484)
            scale_factor = max(scale_factor, 0.581818)
            face_depth = cv2.imread(img_face_depth_name, -1)
            face_depth = np.int32(face_depth)
            # face_depth[face_depth<500] = 500
            face_depth[face_depth>1023] = 1023
            face_depth -= 512
            face_depth = face_depth[np.newaxis, :, :].astype('float') / scale_factor
            sample['face_depth'] = torch.FloatTensor(face_depth / 883)
            sample['scale_factor'] = torch.FloatTensor([scale_factor])
            # print('max: {}, min:{}'.format((face_depth / 430).max(), (face_depth / 430).min()), flush=True)


        if self.kwargs.get('eye_depth'):
            img_lefteye_depth_name = self.img_lefteye_depth_name_list[idx]
            img_righteye_depth_name = self.img_lefteye_depth_name_list[idx]
            le_depth = cv2.imread(img_lefteye_depth_name, -1)
            re_depth = cv2.imread(img_righteye_depth_name, -1)
            le_depth = le_depth[np.newaxis, :, :].astype('float')  # the new dim is the dim with np.newaxis
            re_depth = re_depth[np.newaxis, :, :].astype('float')
            sample['left_depth'] = torch.FloatTensor(le_depth/1000)
            sample['right_depth'] = torch.FloatTensor(re_depth/1000)
        sample['gt'] = torch.FloatTensor(gt)
        if self.kwargs.get('info'):
            # if not self.kwargs.get('eye_depth'):
            img_lefteye_depth_name = self.img_lefteye_depth_name_list[idx]
            img_righteye_depth_name = self.img_lefteye_depth_name_list[idx]
            le_depth = cv2.imread(img_lefteye_depth_name, -1)
            re_depth = cv2.imread(img_righteye_depth_name, -1)
            # get info
            le_depth_ = le_depth[le_depth > 0]
            if len(le_depth_) > 0:
                le_info = [le_coor[0] / 1920, le_coor[1] / 1080, le_depth_.mean() / 1000.]
            else:
                le_info = [le_coor[0] / 1920, le_coor[1] / 1080] + [0.]

            re_depth_ = re_depth[re_depth > 0]
            if len(re_depth_) > 0:
                re_info = [re_coor[0] / 1920, re_coor[1] / 1080, re_depth_.mean() / 1000.]
            else:
                re_info = [re_coor[0] / 1920, re_coor[1] / 1080] + [0.]
            sample['left_info'] = torch.FloatTensor(le_info)
            sample['right_info'] = torch.FloatTensor(re_info)
        if self.kwargs.get('eye_bbx'):
            sample['le_bbx'] = torch.FloatTensor(le_bbx)
            sample['re_bbx'] = torch.FloatTensor(re_bbx)
        if self.kwargs.get('face_bbx'):
            sample['face_bbx'] = torch.FloatTensor(face_bbx)

        if self.kwargs.get('eye_coord'):
            sample['le_coord'] = torch.FloatTensor(np.float32(le_coor))
            sample['re_coord'] = torch.FloatTensor(np.float32(re_coor))
        return sample


class EYEDIAP(Dataset):
    def __init__(self, root, transform=None, split_seed=9363, train=True, **kwargs):
        super(EYEDIAP, self).__init__()
        rnd = Random(split_seed)
        self.transform = transform
        self.kwargs = kwargs
        indeces, self.face = self._scan_dir(os.path.join(root, 'face'))
        _, self.leye = self._scan_dir(os.path.join(root, 'lefteye'))
        _, self.reye = self._scan_dir(os.path.join(root, 'righteye'))
        self.target = pd.read_csv(os.path.join(root, 'gt.txt'), sep=' ', index_col=False, header=None).as_matrix()
        self.headpose = pd.read_csv(os.path.join(root, 'headpose.txt'), sep=' ', index_col=False, header=None).as_matrix().reshape(-1, 3, 3).dot([1, 0, 0])
        self.eye_coord = pd.read_csv(os.path.join(root, 'eye_coordinate.txt'), sep=' ', index_col=False, header=None).as_matrix()
        self.target[:, 0] = self.target[:, 0] / 1920 * 53
        self.target[:, 1] = self.target[:, 1] / 1080 * 30
        self.target = self.target[indeces, ...]
        self.headpose = self.headpose[indeces, ...]
        self.eye_coord = self.eye_coord[indeces, ...]
        # self.train_ids = rnd.choices(range(len(indeces)), k=len(indeces)//5*4)
        self.train_ids = list(range(sum(1 for idx in indeces if idx < 14312)))
        self.val_ids = list(set(range(len(indeces)))-set(self.train_ids))
        if train:
            self.ids = self.train_ids
        else:
            self.ids = self.val_ids

    def _scan_dir(self, dir):
        file_list = {}
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    fid = int(file[:-4])
                    file_list[fid] = os.path.join(root, file)
        indeces = [item[0] for item in sorted(file_list.items(), key=lambda t: t[0])]
        file_list = [item[1] for item in sorted(file_list.items(), key=lambda t: t[0])]
        return indeces, file_list

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample = {}
        id = self.ids[idx]
        face = Image.open(self.face[id]).convert('RGB')
        le = Image.open(self.leye[id]).convert('RGB')
        re = Image.open(self.reye[id]).convert('RGB')

        if self.transform is not None:
            sample['face'] = self.transform(face)
            sample['left_eye'] = self.transform(le)
            sample['right_eye'] = self.transform(re)
        else:
            sample['face'] = face
            sample['left_eye'] = le
            sample['right_eye'] = re

        headpose = self.headpose[id]
        sample['head_pose'] = torch.FloatTensor(headpose.astype('float'))
        sample['gt'] = torch.FloatTensor(self.target[id].astype('float'))
        le_info = self.eye_coord[id][:3]
        re_info = self.eye_coord[id][3:]
        sample['left_info'] = torch.FloatTensor(le_info.astype('float'))
        sample['right_info'] = torch.FloatTensor(re_info.astype('float'))
        return sample

if __name__ == '__main__':
    fire.Fire()