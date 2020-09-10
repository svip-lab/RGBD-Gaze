from gaze_dataset import GazePointAllDataset
from random import Random
from torchvision import transforms
import numpy as np
import cv2
import dlib
from multiprocessing import Pool
from tqdm import tqdm
import pickle
import os
import time


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


data_root = r"D:\data\gaze"

dataset = GazePointAllDataset(root_dir=data_root,
                              transform=None,
                              phase='val',
                              face_image=True,
                              face_bbox=True)
min_depth = []
max_depth = []
mean_depth = []
median_depth = []
bad_samples = []

cnt = 0


def process(i):
    sample = dataset[i]
    face_image = np.asarray(sample["face_image"])
    face_bbox = sample["face_bbox"].numpy()
    w_ori, h_ori = face_bbox[2] - face_bbox[0], face_bbox[3] - face_bbox[1]
    h_now, w_now = face_image.shape[:2]
    scale = np.array(w_ori / w_now, h_ori / h_now)
    pid = f"{sample['pid'].item():05d}"
    sid = f"{sample['sid'].item():05d}"
    os.makedirs(os.path.join(data_root, "landmark", pid), exist_ok=True)
    save_file = os.path.join(data_root, "landmark", pid, sid)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    rects = detector(face_image, 1)
    if len(rects) > 0:
        shape = predictor(face_image, rects[0])
        face_lm = shape_to_np(shape)
        ori_lm = face_lm * scale + face_bbox[:2]
        np.save(save_file + '.npy', ori_lm)
        # Get the landmarks/parts for the face in box d.
        for (x, y) in face_lm:
            cv2.circle(face_image, (x, y), 1, (255, 0, 0), -1)
        cv2.imwrite(save_file + '.jpg', face_image[:, :, ::-1])
        # Draw the face landmarks on the screen.
    else:
        pass
        # print(f"warn: landmark detection failed for pid: {pid}, sid: {sid}", flush=True)
        # np.save(save_file + '.npy', np.zeros((68, 2), dtype="int"))


if __name__ == '__main__':
    with Pool(40) as pool:
        with tqdm(desc="progress", total=len(dataset)) as pbar:
            for i, _ in tqdm(enumerate(pool.imap_unordered(process, range(len(dataset))))):
                pbar.update()
