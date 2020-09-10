from data.gaze_dataset import GazePointAllDataset
from random import Random
from torchvision import transforms
import numpy as np
import cv2
from tqdm import trange
import pickle

rnd = Random()

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

dataset = GazePointAllDataset(root_dir=r"D:\data\gaze",
                              transform=data_transforms['train'],
                              phase='train',
                              face_image=True, face_depth=True, eye_image=True,
                              eye_depth=True,
                              info=True, eye_bbox=True, face_bbox=True, eye_coord=True)

sample = dataset[rnd.choice(range(len(dataset)))]

# face image and depth
face_image, face_depth = sample["face_image"].numpy().transpose((1, 2, 0))[:, :, ::-1], sample["face_depth"].numpy().squeeze()
# cv2.imshow("face_image", face_image)
# cv2.imshow("face_depth", face_depth)
# cv2.waitKey()

# left and right eye image, coord and bbox
left_eye_image, right_eye_image = sample["left_eye_image"].numpy().transpose((1, 2, 0))[:, :, ::-1], sample["right_eye_image"].numpy().transpose((1, 2, 0))[:, :, ::-1],
face_bbox = sample["face_bbox"].numpy()
left_eye_coord, right_eye_coord = sample["left_eye_coord"].numpy(), sample["right_eye_coord"].numpy()
face_scale = sample["face_scale_factor"].item()
left_eye_bbox, right_eye_bbox = sample["left_eye_bbox"].numpy(), sample["right_eye_bbox"].numpy()
left_eye_bbox[:2] -= face_bbox[:2]
left_eye_bbox[2:] -= face_bbox[:2]
right_eye_bbox[:2] -= face_bbox[:2]
right_eye_bbox[2:] -= face_bbox[:2]
left_eye_coord -= face_bbox[:2]
right_eye_coord -= face_bbox[:2]

face_image = (face_image * 255).astype(np.uint8).copy()

cv2.rectangle(face_image, tuple(np.int32(left_eye_bbox[:2] * face_scale).tolist()), tuple(np.int32(left_eye_bbox[2:] * face_scale).tolist()), (0, 255, 0))
cv2.rectangle(face_image, tuple(np.int32(right_eye_bbox[:2] * face_scale).tolist()), tuple(np.int32(right_eye_bbox[2:] * face_scale).tolist()), (0, 255, 0))
cv2.circle(face_image, tuple(np.int32(left_eye_coord * face_scale).tolist()), 3, [0, 0, 255], 1)
cv2.circle(face_image, tuple(np.int32(right_eye_coord * face_scale).tolist()), 3, [0, 0, 255], 1)

cv2.imshow("left_eye_image", left_eye_image)
cv2.imshow("right_eye_image", right_eye_image)
cv2.imshow("face_image", face_image)
cv2.waitKey()

# dataset = GazePointAllDataset(root_dir=r"D:\data\gaze",
#                               transform=None,
#                               phase='train',
#                               face_depth=True)
# min_depth = []
# max_depth = []
# mean_depth = []
# median_depth = []
# bad_samples = []
# for i in trange(len(dataset)):
#     sample = dataset[i]
#     face_depth = sample["face_depth"]
#     if np.sum((face_depth > 0) * (face_depth < 1024)) == 0:
#         bad_samples.append(i)
#         continue
#     median = np.median(face_depth[(face_depth > 0) * (face_depth < 1024)])
#     if not (600 < median < 900):
#         bad_samples.append(i)
#         continue
#     min_depth.append(face_depth[face_depth > 0].min())
#     max_depth.append(face_depth[face_depth < 1024].max())
#     mean_depth.append(face_depth[(face_depth > 0) * (face_depth < 1024)].mean())
#     median_depth.append(median)
#     pass
#
# with open("depth_stat.pkl", "wb+") as fp:
#     pickle.dump((bad_samples, min_depth, max_depth, mean_depth, median_depth), fp)
# print(f"min: {min_depth}, max: {max_depth}\n")
