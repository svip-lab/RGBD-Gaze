from utils.trainer import Trainer
import torch as th
from torch.utils import data
from torch import optim
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import utils.vispyplot as vplt
from data.gaze_dataset_v2 import GazePointAllDataset
import numpy as np
import fire
import logging
import os
import models

th.backends.cudnn.deterministic = False
th.backends.cudnn.benchmark = True


class GazeTrainer(Trainer):
    def __init__(self,
                 # data parameters
                 data_root: str = r'D:\data\gaze',
                 batch_size_train: int = 8,
                 batch_size_val: int = 8,
                 num_workers: int = 20,
                 # trainer parameters
                 is_cuda=True,
                 exp_name="gaze_aaai"
                 ):
        super(GazeTrainer, self).__init__(checkpoint_dir='./ckpt/' + exp_name, is_cuda=is_cuda)
        self.data_root = data_root
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.num_workers = num_workers

        # initialize models
        self.exp_name = exp_name
        model = models.__dict__[exp_name]
        self.models.resnet = model.resnet34(pretrained=True)
        self.models.decoder = model.Decoder()
        self.models.depth_loss = model.DepthL1(0)
        self.models.refine_depth = model.RefineDepth()
        self.weights_init(self.models.decoder)
        self.weights_init(self.models.refine_depth)

        # initialize extra variables
        self.extras.best_loss_base_val = 99999
        self.extras.best_loss_refine_val = 99999
        self.extras.last_epoch_headpose = -1
        self.extras.last_epoch_base = -1

        # initialize meter variables
        self.meters.loss_coord_train = {}
        self.meters.loss_depth_train = {}
        self.meters.loss_coord_val = {}
        self.meters.loss_depth_val = {}
        self.meters.prec_coord_train = {}
        self.meters.prec_coord_val = {}

        # initialize visualizing
        vplt.vis_config.server = 'http://10.10.10.100'
        vplt.vis_config.port = 5000
        vplt.vis_config.env = exp_name

        if os.path.isfile(os.path.join(self.checkpoint_dir, "epoch_latest.pth.tar")):
            self.load_state_dict("epoch_latest.pth.tar")

    def train_base(self, epochs, lr=1e-4, use_refined_depth=False, fine_tune_headpose=True):
        # prepare logger
        self.temps.base_logger = self.logger.getChild('train_base')
        self.temps.base_logger.info('preparing for base training loop.')

        # prepare dataloader
        self.temps.train_loader = self._get_trainloader()
        self.temps.val_loader = self._get_valloader()
        self.temps.num_iters = len(self.temps.train_loader)
        self.temps.lr = lr
        self.temps.epochs = epochs

        self.temps.use_refined_depth = use_refined_depth
        self.temps.fine_tune_headpose = fine_tune_headpose
        # start training loop
        self.temps.epoch = self.extras.last_epoch_base
        self.temps.base_logger.info(f'start base training loop @ epoch {self.extras.last_epoch_base + 1}.')
        for epoch in range(self.extras.last_epoch_base + 1, epochs):
            self.temps.epoch = epoch
            # initialize meters for new epoch
            self._init_base_meters()
            # train one epoch
            self._train_base_epoch()
            # test on validation set
            self._test_base()
            # save checkpoint
            self.extras.last_epoch_base = epoch
            self.save_state_dict(f'epoch_{epoch}.pth.tar')
            self.save_state_dict(f'epoch_latest.pth.tar')
            # plot result
            self._plot_base()
            # logging
            self._log_base()

        # cleaning
        self.models.resnet.cpu()
        self.models.decoder.cpu()
        del self.temps.train_loader
        del self.temps.val_loader

        return self

    def train_headpose(self, epochs, lr=2e-4, lambda_loss_mse=1):
        self.temps.lambda_loss_mse = lambda_loss_mse
        # prepare logger
        self.temps.headpose_logger = self.logger.getChild('train_headpose')
        self.temps.headpose_logger.info('preparing for headpose training loop.')

        # prepare dataloader
        self.temps.train_loader = self._get_trainloader()
        self.temps.val_loader = self._get_valloader()
        self.temps.num_iters = len(self.temps.train_loader)
        self.temps.epochs = epochs
        self.temps.lr = lr

        # start training loop
        self.temps.epoch = self.extras.last_epoch_headpose
        self.temps.headpose_logger.info(
            'start headpose training loop @ epoch {}.'.format(self.extras.last_epoch_headpose + 1))
        for epoch in range(self.extras.last_epoch_headpose + 1, epochs):
            self.temps.epoch = epoch
            # initialize meters for new epoch
            self._init_headpose_meters()
            # train one epoch
            self._train_headpose_epoch()
            # test on validation set
            self._test_headpose()
            # save checkpoint
            self.extras.last_epoch_headpose = epoch
            self.save_state_dict(f'epoch_{epoch}.pth.tar')
            # plot result
            self._plot_headpose()
            # logging
            self._log_headpose()

        # cleaning
        self.models.refine_depth.cpu()
        del self.temps.train_loader
        del self.temps.val_loader

        return self

    def resume(self, filename):
        path = os.path.join(self.checkpoint_dir, filename)
        if os.path.isfile(path):
            self.load_state_dict(filename)
            self.logger.info('load checkpoint from {}'.format(path))
        return self

    def _prepare_model(self, model, train=True):
        if self.is_cuda:
            model.cuda()
        if not isinstance(model, nn.DataParallel):
            model = nn.DataParallel(model)
        if train:
            model.train()
        else:
            model.eval()
        return model

    def _get_trainloader(self):
        logger = self.logger
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

        transformed_train_dataset = GazePointAllDataset(root_dir=self.data_root,
                                                        transform=data_transforms['train'],
                                                        phase='train',
                                                        face_image=True, face_depth=True, eye_image=True,
                                                        eye_depth=True,
                                                        info=True, eye_bbox=True, face_bbox=True, eye_coord=True)
        logger.info('The size of training data is: {}'.format(len(transformed_train_dataset)))
        train_loader = data.DataLoader(transformed_train_dataset, batch_size=self.batch_size_train, shuffle=True,
                                       num_workers=self.num_workers)

        return train_loader

    def _get_valloader(self):
        logger = self.logger
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

        transformed_test_dataset = GazePointAllDataset(root_dir=self.data_root,
                                                       transform=data_transforms['val'],
                                                       phase='val',
                                                       face_image=True, face_depth=True, eye_image=True, eye_depth=True,
                                                       info=True, eye_bbox=True, face_bbox=True, eye_coord=True)

        logger.info('The size of testing data is: {}'.format(len(transformed_test_dataset)))

        test_loader = data.DataLoader(transformed_test_dataset, batch_size=self.batch_size_val, shuffle=False,
                                      num_workers=self.num_workers)
        return test_loader

    def _init_base_meters(self):
        epoch = self.temps.epoch
        self.meters.loss_coord_train[epoch] = []
        self.meters.loss_coord_val[epoch] = 0
        self.meters.prec_coord_train[epoch] = []
        self.meters.prec_coord_val[epoch] = 0

    def _init_headpose_meters(self):
        epoch = self.temps.epoch
        self.meters.loss_depth_train[epoch] = []
        self.meters.loss_depth_val[epoch] = 0

    def _plot_base(self):
        with vplt.set_draw(name='loss_base') as ax:
            ax.plot(list(self.meters.loss_coord_train.keys()),
                    np.mean(list(self.meters.loss_coord_train.values()), axis=1), label='loss_coord_train')
            ax.plot(list(self.meters.loss_coord_val.keys()),
                    list(self.meters.loss_coord_val.values()), label='loss_coord_val')
            ax.set_title('loss base')
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')
            ax.set_xlim(0, self.temps.epochs)
            ax.legend()
            ax.grid(True)
        with vplt.set_draw(name='prec_base') as ax:
            ax.plot(list(self.meters.prec_coord_train.keys()),
                    np.mean(list(self.meters.prec_coord_train.values()), axis=1), label='prec_coord_train')
            ax.plot(list(self.meters.prec_coord_val.keys()),
                    list(self.meters.prec_coord_val.values()), label='prec_coord_val')
            ax.set_title('prec base')
            ax.set_xlabel('epoch')
            ax.set_ylabel('prec')
            ax.set_xlim(0, self.temps.epochs)
            ax.legend()
            ax.grid(True)

    def _plot_headpose(self):
        with vplt.set_draw(name='loss_depth') as ax:
            ax.plot(list(self.meters.loss_depth_train.keys()),
                    np.mean(list(self.meters.loss_depth_train.values()), axis=1), label='loss_depth_train')
            ax.plot(list(self.meters.loss_depth_val.keys()),
                    list(self.meters.loss_depth_val.values()), label='loss_depth_val')
            ax.set_title('loss depth')
            ax.set_xlabel('epoch')
            ax.set_ylabel('loss')
            ax.set_xlim(0, self.temps.epochs)
            ax.legend()
            ax.grid(True)

    def _log_base(self):
        infofmt = "*[{temps.epoch}]\t" \
                  "prec_coord_train: {prec_coord_train:.4f} prec_coord_val: {prec_coord_val:.4f}\t" \
                  "loss_coord_train: {loss_coord_train:.4f} loss_coord_val: {loss_coord_val:.4f}"
        infodict = dict(
            temps=self.temps,
            prec_coord_train=np.mean(self.meters.prec_coord_train[self.temps.epoch]),
            prec_coord_val=np.mean(self.meters.prec_coord_val[self.temps.epoch]),
            loss_coord_train=np.mean(self.meters.loss_coord_train[self.temps.epoch]),
            loss_coord_val=np.mean(self.meters.loss_coord_val[self.temps.epoch]),
        )
        self.temps.base_logger.info(infofmt.format(**infodict))

    def _log_headpose(self):
        infofmt = "*[{temps.epoch}]\t" \
                  "loss_depth_train: {loss_depth_train:.4f} loss_depth_val: {loss_depth_val:.4f}\t"
        infodict = dict(
            temps=self.temps,
            loss_depth_train=np.mean(self.meters.loss_depth_train[self.temps.epoch]),
            loss_depth_val=np.mean(self.meters.loss_depth_val[self.temps.epoch]),
        )
        self.temps.headpose_logger.info(infofmt.format(**infodict))

    def _train_base_epoch(self):
        logger = self.temps.base_logger.getChild('epoch')
        # prepare models
        resnet = self._prepare_model(self.models.resnet)
        decoder = self._prepare_model(self.models.decoder)
        refine_depth = self._prepare_model(self.models.refine_depth, train=True)
        device = th.device("cpu") if not self.is_cuda else th.device("cuda")
        # prepare solvers
        if self.temps.fine_tune_headpose:
            self.temps.base_solver = optim.SGD(self._group_weight(self.models.resnet, lr=self.temps.lr) +
                                               self._group_weight(self.models.decoder, lr=self.temps.lr) +
                                               self._group_weight(self.models.refine_depth, lr=self.temps.lr),
                                               weight_decay=5e-4)
            # self.temps.base_solver = optim.SGD(list(resnet.parameters()) +
            #                                    list(decoder.parameters()) +
            #                                    list(refine_depth.parameters()),
            #                                    lr=self.temps.lr, weight_decay=5e-4)
        else:
            self.temps.base_solver = optim.SGD(self._group_weight(self.models.resnet, lr=self.temps.lr) +
                                               self._group_weight(self.models.decoder, lr=self.temps.lr),
                                               weight_decay=5e-4)
        self.timeit()
        for i, batch in enumerate(self.temps.train_loader):
            self.temps.iter = i
            # prepare data
            face_image, face_depth, face_bbox, \
            left_eye_image, left_eye_depth, left_eye_bbox, left_eye_info, \
            right_eye_image, right_eye_depth, right_eye_bbox, right_eye_info, \
            target, \
            face_factor, left_eye_scale_factor, right_eye_scale_factor = \
                batch['face_image'].to(device), \
                batch['face_depth'].to(device), \
                batch["face_bbox"].to(device), \
                batch["left_eye_image"].to(device), \
                batch["left_eye_depth"].to(device), \
                batch["left_eye_bbox"].to(device), \
                batch["left_eye_info"].to(device), \
                batch["right_eye_image"].to(device), \
                batch["right_eye_depth"].to(device), \
                batch["right_eye_bbox"].to(device), \
                batch["right_eye_info"].to(device), \
                batch["gt"].to(device), \
                batch["face_scale_factor"].to(device), \
                batch["left_eye_scale_factor"].to(device), \
                batch["right_eye_scale_factor"].to(device)

            # measure data loading time
            self.temps.data_time = self._timeit()

            # forward
            lfeat = resnet(left_eye_image)
            rfeat = resnet(right_eye_image)

            if self.temps.fine_tune_headpose:
                head_pose, refined_depth = refine_depth(face_image, face_depth)
            else:
                with th.no_grad():
                    head_pose, refined_depth = refine_depth(face_image, face_depth)

            if self.temps.use_refined_depth:
                with th.no_grad():
                    left_eye_bbox[:, :2] -= face_bbox[:, :2]
                    left_eye_bbox[:, 2:] -= face_bbox[:, :2]
                    right_eye_bbox[:, :2] -= face_bbox[:, :2]
                    right_eye_bbox[:, 2:] -= face_bbox[:, :2]
                    left_eye_bbox = th.clamp(face_factor * left_eye_bbox, min=0, max=223)
                    right_eye_bbox = th.clamp(face_factor * right_eye_bbox, min=0, max=223)

                for j, lb in enumerate(left_eye_bbox):
                    cur_depth = refined_depth[j, :, int(lb[1]):int(lb[3]), int(lb[0]):int(lb[2])]
                    left_eye_info[j, 2] = th.median(cur_depth).item() * face_factor
                for j, rb in enumerate(right_eye_bbox):
                    cur_depth = refined_depth[j, :, int(rb[1]):int(rb[3]), int(rb[0]):int(rb[2])]
                    right_eye_info[j, 2] = th.median(cur_depth).item() * face_factor

            coord = decoder(lfeat, rfeat, head_pose, left_eye_info, right_eye_info)
            loss_coord = F.mse_loss(coord, target)
            prec_coord = self._precision(coord.data, target.data)

            # update resnet & decoder
            self.temps.base_solver.zero_grad()
            loss_coord.backward()
            self.temps.base_solver.step()

            # record loss & accuracy
            epoch = self.temps.epoch
            self.meters.loss_coord_train[epoch].append(loss_coord.item())
            self.meters.prec_coord_train[epoch].append(prec_coord.item())

            # measure batch time
            self.temps.batch_time = self._timeit()

            # logging
            info = f"[{self.temps.epoch}][{self.temps.iter}/{self.temps.num_iters}]\t" \
                   f"data_time: {self.temps.data_time:.2f} batch_time: {self.temps.batch_time:.2f}\t" \
                   f"prec_coord_train: {self.meters.prec_coord_train[self.temps.epoch][-1]:.4f}\t" \
                   f"loss_coord_train: {self.meters.loss_coord_train[self.temps.epoch][-1]:.4f}"
            # infodict = dict(
            #     temps=self.temps,
            #     prec_coord_train=self.meters.prec_coord_train[self.temps.epoch][-1],
            #     loss_coord_train=self.meters.loss_coord_train[self.temps.epoch][-1],
            # )
            logger.info(info)

    def _train_headpose_epoch(self):
        logger = self.temps.headpose_logger.getChild('epoch')
        # prepare models
        refine_depth = self._prepare_model(self.models.refine_depth)
        depth_loss = self.models.depth_loss
        device = th.device("cpu") if not self.is_cuda else th.device("cuda")
        # prepare solvers
        self.temps.headpose_solver = optim.Adam(self._group_weight(self.models.refine_depth, lr=self.temps.lr),
                                                betas=(0.5, 0.999))

        self.timeit()
        for i, batch in enumerate(self.temps.train_loader):
            self.temps.iter = i
            # prepare data
            face_image = batch['face_image'].to(device)
            face_depth = batch['face_depth'].to(device)

            # measure data loading time
            self.temps.data_time = self._timeit()

            # forward
            head_pose, refined_depth = refine_depth(face_image, face_depth)
            loss_depth = depth_loss(refined_depth, face_depth)

            # update resnet & decoder
            self.temps.headpose_solver.zero_grad()
            loss_depth.backward()
            self.temps.headpose_solver.step()

            # record loss & accuracy
            epoch = self.temps.epoch
            self.meters.loss_depth_train[epoch].append(loss_depth.item())

            # measure batch time
            self.temps.batch_time = self._timeit()

            # logging
            infofmt = "[{temps.epoch}][{temps.iter}/{temps.num_iters}]\t" \
                      "data_time: {temps.data_time:.2f} batch_time: {temps.batch_time:.2f}\t" \
                      "loss_depth_train: {loss_depth_train:.4f} "
            infodict = dict(
                temps=self.temps,
                loss_depth_train=self.meters.loss_depth_train[epoch][-1],
            )
            logger.info(infofmt.format(**infodict))

    def _test_base(self):
        logger = self.temps.base_logger.getChild('val')
        resnet = self._prepare_model(self.models.resnet, train=True)
        decoder = self._prepare_model(self.models.decoder, train=True)
        refine_depth = self._prepare_model(self.models.refine_depth, train=True)
        loss_lcoord, loss_rcoord, loss_coord, prec_lcoord, prec_rcoord, prec_coord, num_batches = 0, 0, 0, 0, 0, 0, 0
        device = th.device("cpu") if not self.is_cuda else th.device("cuda")
        for i, batch in enumerate(self.temps.val_loader):
            self.temps.iter = i
            # prepare data
            face_image, face_depth, face_bbox, \
            left_eye_image, left_eye_depth, left_eye_bbox, left_eye_info, \
            right_eye_image, right_eye_depth, right_eye_bbox, right_eye_info, \
            target, \
            face_factor, left_eye_scale_factor, right_eye_scale_factor = \
                batch['face_image'].to(device), \
                batch['face_depth'].to(device), \
                batch["face_bbox"].to(device), \
                batch["left_eye_image"].to(device), \
                batch["left_eye_depth"].to(device), \
                batch["left_eye_bbox"].to(device), \
                batch["left_eye_info"].to(device), \
                batch["right_eye_image"].to(device), \
                batch["right_eye_depth"].to(device), \
                batch["right_eye_bbox"].to(device), \
                batch["right_eye_info"].to(device), \
                batch["gt"].to(device), \
                batch["face_scale_factor"].to(device), \
                batch["left_eye_scale_factor"].to(device), \
                batch["right_eye_scale_factor"].to(device)

            # forward
            with th.no_grad():
                lfeat = resnet(left_eye_image)
                rfeat = resnet(right_eye_image)
                head_pose, refined_depth = refine_depth(face_image, face_depth)

                left_eye_bbox[:, :2] -= face_bbox[:, :2]
                left_eye_bbox[:, 2:] -= face_bbox[:, :2]
                right_eye_bbox[:, :2] -= face_bbox[:, :2]
                right_eye_bbox[:, 2:] -= face_bbox[:, :2]
                left_eye_bbox = th.clamp(face_factor * left_eye_bbox, min=0, max=223)
                right_eye_bbox = th.clamp(face_factor * right_eye_bbox, min=0, max=223)

                if self.temps.use_refined_depth:
                    for j, lb in enumerate(left_eye_bbox):
                        cur_depth = refined_depth[j, :, int(lb[1]):int(lb[3]), int(lb[0]):int(lb[2])]
                        left_eye_info[j, 2] = th.median(cur_depth).item() * face_factor
                    for j, rb in enumerate(right_eye_bbox):
                        cur_depth = refined_depth[j, :, int(rb[1]):int(rb[3]), int(rb[0]):int(rb[2])]
                        right_eye_info[j, 2] = th.median(cur_depth).item() * face_factor

                coord = decoder(lfeat, rfeat, head_pose, left_eye_info, right_eye_info)
                loss_coord_iter = F.mse_loss(coord, target)
                prec_coord_iter = self._precision(coord.data, target.data)

            # accumulate meters
            loss_coord += loss_coord_iter.item()
            prec_coord += prec_coord_iter.item()
            num_batches += 1
            # logging
            infofmt = "[{temps.epoch}]\t" \
                      "prec_coord: {prec_coord: .4f}\t" \
                      "loss_coord: {loss_coord: .4f}"
            infodict = dict(
                temps=self.temps,
                loss_coord=loss_coord_iter,
                prec_coord=prec_coord_iter
            )
            logger.info(infofmt.format(**infodict))

        # record meters
        epoch = self.temps.epoch
        self.meters.loss_coord_val[epoch] = loss_coord / num_batches
        self.meters.prec_coord_val[epoch] = prec_coord / num_batches

    def _test_headpose(self):
        logger = self.temps.headpose_logger.getChild('val')
        refine_depth = self._prepare_model(self.models.refine_depth)
        depth_loss = self.models.depth_loss
        loss_depth, num_batchs = 0, 0
        device = th.device("cpu") if not self.is_cuda else th.device("cuda")
        for i, batch in enumerate(self.temps.val_loader):
            self.temps.iter = i
            # prepare data
            face_image = batch['face_image'].to(device)
            face_depth = batch['face_depth'].to(device)

            # measure data loading time
            self.temps.data_time = self._timeit()

            # forward
            with th.no_grad():
                head_pose, refined_depth = refine_depth(face_image, face_depth)
                loss_depth_iter = depth_loss(refined_depth, face_depth)

            # accumulate meters
            loss_depth += loss_depth_iter.item()
            num_batchs += 1
            # logging
            infofmt = "[{temps.epoch}]\t" \
                      "loss_depth: {loss_depth:.4f}"
            infodict = dict(
                temps=self.temps,
                loss_depth=loss_depth_iter,
            )
            logger.info(infofmt.format(**infodict))

        # record meters
        epoch = self.temps.epoch
        self.meters.loss_depth_val[epoch] = loss_depth / num_batchs

    @staticmethod
    def _precision(out, target):
        return th.mean(th.sqrt(th.sum((out - target) ** 2, 1)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='<%(name)s:%(levelname)s> %(message)s')
    fire.Fire(GazeTrainer)
