import torch as th
from torch.nn import functional as F
import numpy as np
import cvxpy as cp
import visdom
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt

print(f"matplotlib version: {matplotlib.__version__}")

# initialize dataset
from torchvision import transforms
from data.gaze_dataset_v2 import GazePointAllDataset
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
trainset = GazePointAllDataset(root_dir=r'D:\data\gaze',
                               transform=data_transforms['train'],
                               phase='train',
                               face_image=True, face_depth=True, eye_image=True,
                               eye_depth=True,
                               info=True, eye_bbox=True, face_bbox=True, eye_coord=True,
                               landmark=True)
print('The size of training data is: {}'.format(len(trainset)))


def scatter_face_landmark(face_landmark, median, median_mask=None):
    if median_mask is None:
        median_mask = np.ones_like(median, dtype=np.int)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('D')
    xs = face_landmark[median_mask, 0]
    ys = face_landmark[median_mask, 1]
    zs = median[median_mask]
    ax.scatter(xs, ys, zs, c="r", marker="x")
    ax.view_init(-109, -64)
    plt.draw()
    plt.show()


def extract_landmark_depth(depth, landmark, scale_factor, bbox, region_size=7):
    bs = depth.size(0)
    img_size = depth.size(3)
    assert depth.size(2) == depth.size(3)
    num_landmark = landmark.size(1)
    # transform landmarks to face image coordinate system (bs x lm x 2)
    face_lm_image = (landmark - bbox[:, :2].unsqueeze(1).to(depth)) * scale_factor.unsqueeze(1)
    face_lm = (face_lm_image / img_size * 2) - 1.

    # sample landmark region (bs x lm x lm_size x lm_size)
    # gen sample grid (bs x lm x lm_size x lm_size x 2)
    x = th.linspace(-region_size / 2, region_size / 2, region_size) / img_size * 2
    grid = th.stack(th.meshgrid([x, x])[::-1], dim=2).to(depth)
    grid = face_lm.view(bs, num_landmark, 1, 1, 2) + grid
    depth_landmark_regions = F.grid_sample(
        depth, grid.view(bs, num_landmark, -1, 2), mode="nearest", padding_mode="zeros"
    ).squeeze(1)

    # non-zero median
    depth_landmark_regions_sorted = th.sort(depth_landmark_regions, dim=2)[0]
    depth_landmark_regions_mask = depth_landmark_regions_sorted > 1.e-4
    depth_landmark_regions_mask[:, :, 0] = 0
    depth_landmark_regions_mask[:, :, 1:] = depth_landmark_regions_mask[:, :, 1:] - \
                                            depth_landmark_regions_mask[:, :, :-1]
    depth_landmark_regions_mask[:, :, -1] = ((depth_landmark_regions_mask.sum(dim=2) == 0) + depth_landmark_regions_mask[:, :, -1]) > 0
    assert (depth_landmark_regions_mask.sum(dim=2) == 1).all(), f"{th.sum(depth_landmark_regions_mask, dim=2)}\n{depth_landmark_regions_mask[:, :, depth_landmark_regions_mask.size(2) - 1]}\n{depth_landmark_regions_mask.sum(dim=2) == 0}"
    nonzero_st = th.nonzero(depth_landmark_regions_mask)

    assert (nonzero_st[1:, 0] - nonzero_st[:-1, 0] >= 0).all() and \
           ((nonzero_st[1:, 0] * num_landmark + nonzero_st[1:, 1]) -
            (nonzero_st[:-1, 0] * num_landmark + nonzero_st[:-1, 1]) >= 0).all()
    assert nonzero_st.size(0) == bs * num_landmark
    median_ind = ((nonzero_st[:, 2] + region_size * region_size - 1) / 2).long()
    depth_landmark_regions_sorted = depth_landmark_regions_sorted.view(bs * num_landmark, region_size * region_size)
    median = depth_landmark_regions_sorted[range(len(median_ind)), median_ind].view(bs, num_landmark)
    median_mask = median > 1.e-4

    return median, median_mask, face_lm_image


def median_to_rel(median, median_mask, landmark):
    rel_median = median.view(68, 1).expand(68, 68) - median.view(1, 68).expand(68, 68)
    landmark_dist = th.norm(landmark.view(68, 1, 2).expand(68, 68, 2) - landmark.view(1, 68, 2).expand(68, 68, 2), dim=2)
    rel_median = rel_median / (landmark_dist + 1e-4)
    rel_median_mask = median_mask.view(68, 1).expand(68, 68) * median_mask.view(1, 68).expand(68, 68)
    return rel_median, rel_median_mask, landmark_dist


sample = trainset[3917]
face_image, face_depth, face_bbox, face_factor, face_landmark = \
    sample['face_image'], \
    sample['face_depth'], \
    sample["face_bbox"], \
    sample["face_scale_factor"], \
    sample["face_landmark"]

median, median_mask, face_lm_image = extract_landmark_depth(face_depth.unsqueeze(0), face_landmark.unsqueeze(0),
                                                            face_factor.unsqueeze(0), face_bbox.unsqueeze(0), 7)
median = median[0].numpy() * 500 + 500
median_mask = np.abs(median - np.median(median)) < 90
rel_median, rel_median_mask, landmark_dist = median_to_rel(th.from_numpy(median),
                                                           th.from_numpy(median_mask.astype("float")), face_landmark)
rel_median, rel_median_mask, landmark_dist = rel_median.numpy(), rel_median_mask.numpy(), landmark_dist.numpy()
face_lm_image = face_lm_image[0].numpy()
scatter_face_landmark(face_lm_image, median)
# input("Press Enter to continue...")

