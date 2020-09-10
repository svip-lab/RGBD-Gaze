from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet, model_urls
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter
import torch as th
import pdb
import cv2
import numpy as np


class ResNetEncoder(ResNet):
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x112_64 = x
        x = self.maxpool(x)
        x = self.layer1(x)
        # x56_64 = x
        x = self.layer2(x)
        # x28_128 = x
        x = self.layer3(x)
        # x14_256 = x
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(x)

        return x  # , x112_64, x56_64, x28_128, x14_256


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetEncoder(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetEncoder(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetEncoder(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetEncoder(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetEncoder(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


class Depth2AbsDepth(nn.Module):
    pass


class RGBD2AbsDepth(nn.Module):
    pass


class RGB2AbsDepth(nn.Module):
    pass


class Depth2RelDepth(nn.Module):
    pass


def extract_landmark_depth(depth, landmark, scale_factor, bbox, region_size=7, depth_scale=500.,
                           valid_depth_range=90, debug=False):
    bs = depth.size(0)
    img_size = depth.size(3)
    assert depth.size(2) == depth.size(3)
    num_landmark = landmark.size(1)
    # transform landmarks to face image coordinate system (bs x lm x 2)
    face_lm = ((landmark - bbox[:, :2].unsqueeze(1).to(depth)) * scale_factor.unsqueeze(1) / img_size * 2) - 1.

    # sample landmark region (bs x lm x lm_size x lm_size)
    # gen sample grid (bs x lm x lm_size x lm_size x 2)
    x = th.linspace(-region_size / 2, region_size / 2, region_size) / img_size * 2
    grid = th.stack(th.meshgrid([x, x])[::-1], dim=2).to(depth)
    grid = face_lm.view(bs, num_landmark, 1, 1, 2) + grid
    depth_landmark_regions = F.grid_sample(
        depth, grid.view(bs, num_landmark, -1, 2), mode="nearest", padding_mode="zeros"
    ).squeeze(1)

    if debug:
        # visualize landmark
        for dep, lms, lmbs in zip(depth, face_lm, grid):
            depth_vis = np.uint8(dep.squeeze(0).cpu().numpy() * 255)
            depth_vis = np.stack([depth_vis, depth_vis, depth_vis], axis=2)
            for lm, lmb in zip(lms, lmbs):
                cv2.circle(depth_vis, tuple(((lm + 1) * 112).long().tolist()), 5, (0, 0, 255), 2)
                x1, y1 = int((lmb[0, 0, 0].item() + 1) * 112), int((lmb[0, 0, 1].item() + 1) * 112)
                x2, y2 = int((lmb[region_size-1, region_size-1, 0].item() + 1) * 112), int((lmb[region_size-1, region_size-1, 1].item() + 1) * 112)
                cv2.rectangle(depth_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("res", depth_vis)
            cv2.waitKey()

    # non-zero median if exists, else return zero
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
    median_mask = th.abs(median - th.median(median, dim=1)[0].unsqueeze(1)) < (valid_depth_range / depth_scale)

    return median, median_mask


class RelMatrix2Col(nn.Module):
    def __init__(self, num_landmark=68):
        super(RelMatrix2Col, self).__init__()
        xx = th.arange(0, num_landmark).unsqueeze(1).expand(num_landmark, num_landmark)
        yy = xx.t()
        self.register_buffer("col_mask", xx > yy)

    def forward(self, rel_matrix):
        assert th.allclose(rel_matrix, -rel_matrix.transpose(rel_matrix.dim() - 1, rel_matrix.dim() - 2))
        return rel_matrix[..., self.col_mask]


class RelCol2Matrix(nn.Module):
    def __init__(self, num_landmark=68):
        super(RelCol2Matrix, self).__init__()
        self.num_landmark = num_landmark
        xx = th.arange(0, num_landmark).unsqueeze(1).expand(num_landmark, num_landmark)
        yy = xx.t()
        self.register_buffer("col_mask", xx > yy)

    def forward(self, rel_column):
        rel_matrix = rel_column.new_full((rel_column.size(0), self.num_landmark, self.num_landmark), 0)
        rel_matrix[..., self.col_mask] = rel_column
        rel_matrix[..., 1 - self.col_mask] = - rel_matrix.transpose(
            rel_matrix.dim() - 1, rel_matrix.dim() - 2
        )[..., 1 - self.col_mask]
        assert th.allclose(rel_matrix, -rel_matrix.transpose(rel_matrix.dim() - 1, rel_matrix.dim() - 2))
        return rel_matrix


class RGB2RelDepth(nn.Module):
    def __init__(self, num_landmark=68, dim_landmark_feat=128, initial_bias=None):
        super(RGB2RelDepth, self).__init__()
        self.encoder = resnet18(pretrained=True)
        self.num_landmark = num_landmark
        self.dim_rel_col = int(num_landmark * (num_landmark - 1) // 2)
        self.dim_embedding = dim_landmark_feat
        self.rel_depth = nn.Sequential(
            nn.Linear(512, self.dim_rel_col, bias=True),
            RelCol2Matrix(num_landmark=num_landmark)
        )
        if initial_bias is not None:
            nn.init.constant_(self.rel_depth[0].bias, initial_bias)

    def forward(self, face_image):
        feat = self.encoder(face_image)
        rel_depth = self.rel_depth(feat)

        return rel_depth


class LossRelDepth(nn.Module):
    def __init__(self, crit, num_landmark=68, image_size=224, landmark_region_size=7, depth_scale=500.,
                 valid_depth_range=90):
        super(LossRelDepth, self).__init__()
        self.crit = crit
        self.num_landmark = num_landmark
        self.image_size = image_size
        self.lm_region_size = landmark_region_size
        self.depth_scale = depth_scale
        self.valid_depth_range = valid_depth_range
        xx = th.arange(0, num_landmark).unsqueeze(1).expand(num_landmark, num_landmark)
        self.register_buffer("diag_mask", xx == xx.t())

    def repeat_as_col(self, col):
        return col.view(
            col.size(0), self.num_landmark, 1, -1
        ).expand(
            col.size(0), self.num_landmark, self.num_landmark, -1
        ).squeeze()

    def repeat_as_row(self, row):
        return row.view(
            row.size(0), 1, self.num_landmark, -1
        ).expand(
            row.size(0), self.num_landmark, self.num_landmark, -1
        ).squeeze()

    def forward(self, rel_depth_pred, depth, landmarks, scale_factor, bbox):
        bs = rel_depth_pred.size(0)
        with th.no_grad():
            landmark_dist = th.norm(self.repeat_as_col(landmarks) - self.repeat_as_row(landmarks), dim=3)
            assert th.allclose(landmark_dist, landmark_dist.transpose(1, 2))
            median, median_mask = extract_landmark_depth(
                depth=depth,
                landmark=landmarks,
                scale_factor=scale_factor,
                bbox=bbox,
                region_size=self.lm_region_size,
                depth_scale=self.depth_scale,
                valid_depth_range=self.valid_depth_range
            )
            median *= self.depth_scale
            median_diff = (self.repeat_as_col(median) - self.repeat_as_row(median)) #/ scale_factor.view(bs, 1, 1)
            median_rel_mask = (self.repeat_as_col(median_mask) * self.repeat_as_row(median_mask)).to(rel_depth_pred)
            # assert (median_diff[..., self.diag_mask] == 0).all()
            assert th.allclose(median_diff, -median_diff.transpose(1, 2))
        loss_ele = self.crit(rel_depth_pred, median_diff, reduction='none')
        loss = th.sum(loss_ele * median_rel_mask) / (th.sum(median_rel_mask) + 1e-4)

        return loss, median_diff, median_rel_mask


# class RGB2RelDepth(nn.Module):
#     pass
