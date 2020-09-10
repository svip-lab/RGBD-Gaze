from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet, model_urls
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import functional as F
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


def extract_landmark_depth(depth, landmark, scale_factor, bbox, region_size=7):
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

    # while True:
    #     # visualize landmark
    #     for dep, lms, lmbs in zip(depth, face_lm, grid):
    #         depth_vis = np.uint8(dep.squeeze(0).cpu().numpy() * 255)
    #         depth_vis = np.stack([depth_vis, depth_vis, depth_vis], axis=2)
    #         for lm, lmb in zip(lms, lmbs):
    #             cv2.circle(depth_vis, tuple(((lm + 1) * 112).long().tolist()), 5, (0, 0, 255), 2)
    #             x1, y1 = int((lmb[0, 0, 0].item() + 1) * 112), int((lmb[0, 0, 1].item() + 1) * 112)
    #             x2, y2 = int((lmb[region_size-1, region_size-1, 0].item() + 1) * 112), int((lmb[region_size-1, region_size-1, 1].item() + 1) * 112)
    #             cv2.rectangle(depth_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         cv2.imshow("res", depth_vis)
    #         cv2.waitKey()
    #     break

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

    return median, median_mask


class RGB2RelDepth(nn.Module):
    def __init__(self, num_landmark=68):
        super(RGB2RelDepth, self).__init__()
        self.encoder = resnet18(pretrained=True)
        self.num_landmark = num_landmark
        self.depth = nn.Sequential(
            nn.Linear(512, num_landmark, bias=True),
            # nn.Sigmoid()
        )

    def forward(self, face_image, landmarks):
        bs = face_image.size(0)
        feat = self.encoder(face_image)
        depth = self.depth(feat)
        d1 = depth.view(bs, self.num_landmark, 1).expand(bs, self.num_landmark, self.num_landmark)
        d2 = depth.view(bs, 1, self.num_landmark).expand(bs, self.num_landmark, self.num_landmark)
        depthdiff = d1 - d2
        assert th.allclose(depthdiff, -depthdiff.transpose(1, 2))
        with th.no_grad():
            lm1 = landmarks.view(bs, self.num_landmark, 1, 2).expand(bs, self.num_landmark, self.num_landmark, 2)
            lm2 = landmarks.view(bs, 1, self.num_landmark, 2).expand(bs, self.num_landmark, self.num_landmark, 2)
            lmdist = th.norm((lm1 - lm2).to(feat), dim=3)
            assert th.allclose(lmdist, lmdist.transpose(1, 2))
        rel_depth = depthdiff / (lmdist + 1e-4)

        return rel_depth


class LossRelDepth(nn.Module):
    def __init__(self, crit, num_landmark=68, image_size=224, landmark_region_size=7, depth_scale=500):
        super(LossRelDepth, self).__init__()
        self.crit = crit
        self.num_landmark = num_landmark
        self.image_size = image_size
        self.lm_region_size = landmark_region_size
        self.depth_scale = depth_scale

    def forward(self, rel_depth_pred, depth, landmarkds, scale_factor, bbox):
        with th.no_grad():
            median, median_mask = extract_landmark_depth(depth, landmarkds, scale_factor, bbox, self.lm_region_size)
            median *= self.depth_scale
            bs = rel_depth_pred.size(0)
            median_rel_mask = median_mask.view(bs, self.num_landmark, 1).expand(bs, self.num_landmark,
                                                                                self.num_landmark) * \
                              median_mask.view(bs, 1, self.num_landmark).expand(bs, self.num_landmark,
                                                                                self.num_landmark)
            diag_mask = 1 - th.eye(self.num_landmark, self.num_landmark).to(rel_depth_pred)
            rel_median = median.view(bs, self.num_landmark, 1).expand(bs, self.num_landmark, self.num_landmark) - \
                         median.view(bs, 1, self.num_landmark).expand(bs, self.num_landmark, self.num_landmark)
            landmark_dist = th.norm(
                landmarkds.view(bs, self.num_landmark, 1, 2).expand(bs, self.num_landmark, self.num_landmark, 2) -
                landmarkds.view(bs, 1, self.num_landmark, 2).expand(bs, self.num_landmark, self.num_landmark, 2),
                dim=3
            )
            assert th.allclose(landmark_dist, landmark_dist.transpose(1, 2))
            rel_median = rel_median / (landmark_dist + 1e-4) * diag_mask
        loss = th.sum(self.crit(rel_depth_pred, rel_median, reduce=False) * median_rel_mask.to(median)) / \
               (th.sum(median_rel_mask) + 1e-4)

        return loss


# class RGB2RelDepth(nn.Module):
#     pass
