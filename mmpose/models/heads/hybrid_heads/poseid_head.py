# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import torch
from mmcv.cnn import build_conv_layer, build_upsample_layer
from mmengine.structures import PixelData
from torch import Tensor, nn

from mmpose.evaluation.functional import pose_pck_accuracy
from mmpose.models.utils.tta import flip_heatmaps
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions)
from ..base_head import BaseHead

import numpy as np

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class PoseIDHead(BaseHead):
    """Multi-variate head predicting all information about keypoints. Apart 
    from the heatmap, it also predicts:
        1) Heatmap for each keypoint
        2) Usefulness of the pose for identification
    The heatmap predicting part is the same as HeatmapHead introduced in
    in `Simple Baselines`_ by Xiao et al (2018).

    Args:
        in_channels (int | Sequence[int]): Number of channels in the input
            feature map
        out_channels (int): Number of channels in the output heatmap
        deconv_out_channels (Sequence[int], optional): The output channel
            number of each deconv layer. Defaults to ``(256, 256, 256)``
        deconv_kernel_sizes (Sequence[int | tuple], optional): The kernel size
            of each deconv layer. Each element should be either an integer for
            both height and width dimensions, or a tuple of two integers for
            the height and the width dimension respectively.Defaults to
            ``(4, 4, 4)``
        conv_out_channels (Sequence[int], optional): The output channel number
            of each intermediate conv layer. ``None`` means no intermediate
            conv layer between deconv layers and the final conv layer.
            Defaults to ``None``
        conv_kernel_sizes (Sequence[int | tuple], optional): The kernel size
            of each intermediate conv layer. Defaults to ``None``
        final_layer_dict (dict): Arguments of the final Conv2d layer.
            Defaults to ``dict(kernel_size=1)``
        keypoint_loss (Config): Config of the keypoint loss. Defaults to use
            :class:`KeypointMSELoss`
        usefulness_loss (Config): Config of the probability loss. Defaults to use
            :class:`BCELoss`
        freeze_heatmaps (bool): Whether to freeze the heatmaps prediction.
            Defaults to ``False``
        freeze_usefulness (bool): Whether to freeze the usefulness prediction.
            Defaults to ``False``
        decoder (Config, optional): The decoder config that controls decoding
            keypoint coordinates from the network output. Defaults to ``None``
        init_cfg (Config, optional): Config to control the initialization. See
            :attr:`default_init_cfg` for default settings


    .. _`Simple Baselines`: https://arxiv.org/abs/1804.06208
    """

    _version = 2

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 out_channels: int,
                 deconv_out_channels: OptIntSeq = (256, 256, 256),
                 deconv_kernel_sizes: OptIntSeq = (4, 4, 4),
                 conv_out_channels: OptIntSeq = None,
                 conv_kernel_sizes: OptIntSeq = None,
                 final_layer_dict: dict = dict(kernel_size=1),
                 keypoint_loss: ConfigType = dict(
                     type='KeypointMSELoss', use_target_weight=True),
                 usefulness_loss: ConfigType = dict(
                     type='MSELoss', use_target_weight=True),
                 usefulness_thr: float = None,
                 freeze_heatmaps: bool = False,
                 freeze_usefulness: bool = False,
                 detach_usefulness: bool = True,
                 decoder: OptConfigType = dict(
                    type='UDPHeatmap', input_size=(192, 256),
                    heatmap_size=(48, 64), sigma=2),
                 init_cfg: OptConfigType = None,
        ):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.keypoint_loss_module = MODELS.build(keypoint_loss)
        self.usefulness_loss_module = MODELS.build(usefulness_loss)

        self.decoder = KEYPOINT_CODECS.build(decoder)
        self.nonlinearity = nn.ReLU(inplace=True)
        
        self.usefulness_thr = usefulness_thr
        self.detach_usefulness = detach_usefulness

        self._build_heatmap_head(
            in_channels=in_channels,
            out_channels=out_channels,
            deconv_out_channels=deconv_out_channels,
            deconv_kernel_sizes=deconv_kernel_sizes,
            conv_out_channels=conv_out_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            final_layer_dict=final_layer_dict,
            normalize=False,
            freeze=freeze_heatmaps)
        
        self._build_usefulness_head(
            in_channels=in_channels,
            out_channels=out_channels,
            freeze=freeze_usefulness)
        
        # Register the hook to automatically convert old version state dicts
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def _build_heatmap_head(self, in_channels: int, out_channels: int,
                            deconv_out_channels: Sequence[int],
                            deconv_kernel_sizes: Sequence[int],
                            conv_out_channels: Sequence[int],
                            conv_kernel_sizes: Sequence[int],
                            final_layer_dict: dict,
                            normalize: bool = False,
                            freeze: bool = False) -> nn.Module:
        """Build the heatmap head module."""
        if deconv_out_channels:
            if deconv_kernel_sizes is None or len(deconv_out_channels) != len(
                    deconv_kernel_sizes):
                raise ValueError(
                    '"deconv_out_channels" and "deconv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {deconv_out_channels} and '
                    f'{deconv_kernel_sizes}')

            self.deconv_layers = self._make_deconv_layers(
                in_channels=in_channels,
                layer_out_channels=deconv_out_channels,
                layer_kernel_sizes=deconv_kernel_sizes,
            )
            in_channels = deconv_out_channels[-1]
        else:
            self.deconv_layers = nn.Identity()

        if conv_out_channels:
            if conv_kernel_sizes is None or len(conv_out_channels) != len(
                    conv_kernel_sizes):
                raise ValueError(
                    '"conv_out_channels" and "conv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {conv_out_channels} and '
                    f'{conv_kernel_sizes}')

            self.conv_layers = self._make_conv_layers(
                in_channels=in_channels,
                layer_out_channels=conv_out_channels,
                layer_kernel_sizes=conv_kernel_sizes)
            in_channels = conv_out_channels[-1]
        else:
            self.conv_layers = nn.Identity()

        if final_layer_dict is not None:
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1)
            cfg.update(final_layer_dict)
            self.final_layer = build_conv_layer(cfg)
        else:
            self.final_layer = nn.Identity()
        self.normalize_layer = nn.Sigmoid() if normalize else nn.Identity()

        if freeze:
            for param in self.deconv_layers.parameters():
                param.requires_grad = False
            for param in self.conv_layers.parameters():
                param.requires_grad = False
            for param in self.final_layer.parameters():
                param.requires_grad = False

    def _build_usefulness_head(self, in_channels: int, out_channels: int,
                                freeze: bool = False) -> nn.Module:
        """Build the probability head module."""
        usf_layers = []
        kernel_sizes = [(4, 3), (2, 2), (2, 2)]
        for i in range(len(kernel_sizes)):
            usf_layers.append(
                build_conv_layer(
                    dict(type='Conv2d'),
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            usf_layers.append(
                nn.BatchNorm2d(num_features=in_channels))
            usf_layers.append(
                nn.MaxPool2d(kernel_size=kernel_sizes[i], stride=kernel_sizes[i], padding=0))
            usf_layers.append(self.nonlinearity)
        usf_layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=in_channels,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0))
        usf_layers.append(nn.Sigmoid())
        self.usefulness_layers = nn.Sequential(*usf_layers)

        if freeze:
            for param in self.usefulness_layers.parameters():
                param.requires_grad = False

    def _make_conv_layers(self, in_channels: int,
                          layer_out_channels: Sequence[int],
                          layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create convolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            padding = (kernel_size - 1) // 2
            cfg = dict(
                type='Conv2d',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding)
            layers.append(build_conv_layer(cfg))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(self.nonlinearity)
            in_channels = out_channels

        return nn.Sequential(*layers)

    def _make_deconv_layers(self, in_channels: int,
                            layer_out_channels: Sequence[int],
                            layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create deconvolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise ValueError(f'Unsupported kernel size {kernel_size} for'
                                 'deconvlutional layers in '
                                 f'{self.__class__.__name__}')
            cfg = dict(
                type='deconv',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False)
            layers.append(build_upsample_layer(cfg))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(self.nonlinearity)
            in_channels = out_channels

        return nn.Sequential(*layers)

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(
                type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1)
        ]
        return init_cfg

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward the network. The input is multi scale feature maps and the
        output is (1) the heatmap, (2) probability, (3) visibility, (4) oks and (5) error.

        Args:
            feats (Tensor): Multi scale feature maps.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: outputs.
        """
        x = feats[-1]

        heatmaps = self.forward_heatmap(x)
        usefulness = self.forward_usefulness(x)
        
        return heatmaps, usefulness
    
    def forward_heatmap(self, x: Tensor) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the heatmap.

        Args:
            x (Tensor): Multi scale feature maps.

        Returns:
            Tensor: output heatmap.
        """
        x = self.deconv_layers(x)
        x = self.conv_layers(x)
        x = self.final_layer(x)
        x = self.normalize_layer(x)
        return x
    
    def forward_usefulness(self, x: Tensor) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the probability.

        Args:
            x (Tensor): Multi scale feature maps.
            detach (bool): Whether to detach the probability from gradient

        Returns:
            Tensor: output probability.
        """
        if self.detach_usefulness:
            x = x.detach()
        x = self.usefulness_layers(x)
        return x

    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:
        """Predict results from features.

        Args:
            feats (Tuple[Tensor] | List[Tuple[Tensor]]): The multi-stage
                features (or multiple multi-stage features in TTA)
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            test_cfg (dict): The runtime config for testing process. Defaults
                to {}

        Returns:
            Union[InstanceList | Tuple[InstanceList | PixelDataList]]: If
            ``test_cfg['output_heatmap']==True``, return both pose and heatmap
            prediction; otherwise only return the pose prediction.

            The pose prediction is a list of ``InstanceData``, each contains
            the following fields:

                - keypoints (np.ndarray): predicted keypoint coordinates in
                    shape (num_instances, K, D) where K is the keypoint number
                    and D is the keypoint dimension
                - keypoint_scores (np.ndarray): predicted keypoint scores in
                    shape (num_instances, K)

            The heatmap prediction is a list of ``PixelData``, each contains
            the following fields:

                - heatmaps (Tensor): The predicted heatmaps in shape (K, h, w)
        """

        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats

            _htm, _usf = self.forward(_feats)
            _htm_flip, _usf_flip = self.forward(_feats_flip)
            B, C, H, W = _htm.shape

            # Flip back the keypoints
            _htm_flip = flip_heatmaps(
                _htm_flip,
                flip_mode=test_cfg.get('flip_mode', 'heatmap'),
                flip_indices=flip_indices,
                shift_heatmap=test_cfg.get('shift_heatmap', False))
            heatmaps = (_htm + _htm_flip) * 0.5

            # Flip back scalars
            # _usf_flip = _usf_flip[:, flip_indices]
            
            usefulness = (_usf + _usf_flip) * 0.5
        else:
            heatmaps, usefulness = self.forward(feats)
            B, C, H, W = heatmaps.shape

        preds = self.decode(heatmaps)
        usefulness = to_numpy(usefulness).reshape((B, 1))
        
        for pi, p in enumerate(preds):
            p.set_field(usefulness[pi], "keypoints_usf")
            
        if test_cfg.get('output_heatmaps', False):
            pred_fields = [
                PixelData(heatmaps=hm) for hm in heatmaps.detach()
            ]
            return preds, pred_fields
        else:
            return preds

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            feats (Tuple[Tensor]): The multi-stage features
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            train_cfg (dict): The runtime config for training process.
                Defaults to {}

        Returns:
            dict: A dictionary of losses.
        """
        dt_heatmaps, dt_usfs = self.forward(feats)
        device=dt_heatmaps.device
        B, C, H, W = dt_heatmaps.shape
        
        # Extract GT data
        gt_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in batch_data_samples])
        # breakpoint()
        gt_usfs = np.stack(
            [d.gt_instances.identified.astype(float) for d in batch_data_samples])
        if self.usefulness_thr is not None:
            gt_usfs = (gt_usfs > self.usefulness_thr).astype(int)

        gt_annotated = np.stack(
            [d.gt_instances.keypoints_visible.astype(int) for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])

        # Convert everything to tensors
        gt_usfs = torch.tensor(gt_usfs, device=device, dtype=dt_usfs.dtype)
        gt_annotated = torch.tensor(gt_annotated, device=device)
        
        # Reshape everything to comparable shapes
        gt_heatmaps = gt_heatmaps.view((B, C, H, W))
        dt_heatmaps = dt_heatmaps.view((B, C, H, W))
        gt_usfs = gt_usfs.view((B, 1))
        dt_usfs = dt_usfs.view((B, 1))
        keypoint_weights = keypoint_weights.view((B, C))
        gt_annotated = gt_annotated.view((B, C))

        # Compute uselfulness weights
        # usfs_weights = torch.ones_like(dt_usfs, dtype=torch.float, device=device)
        usfs_weights = gt_usfs.detach().clone() * 8.0 + 1.0     # Weight the useful poses more ais the ratio in data is approx 1:9

        # calculate losses
        losses = dict()
        heatmap_weights = keypoint_weights

        heatmap_loss     = self.keypoint_loss_module(dt_heatmaps, gt_heatmaps, heatmap_weights)
        usefulness_loss  = self.usefulness_loss_module(
            dt_usfs, gt_usfs,
            target_weight=usfs_weights
        )
        
        losses.update(
            loss_kpt=heatmap_loss,
            loss_usefulness=usefulness_loss,
        )
        
        # calculate accuracy
        if train_cfg.get('compute_acc', True):
            acc_pose = self.get_pose_accuracy(
                dt_heatmaps, gt_heatmaps, keypoint_weights > 0.5
            )
            losses.update(acc_pose=acc_pose)

            # Calculate the best binary accuracy for probability
            if self.usefulness_thr is not None:
                usf_acc, usf_thr = self.get_binary_accuracy(
                    dt_usfs, gt_usfs, torch.ones_like(dt_usfs, dtype=torch.bool)
                )
                losses.update(usf_acc=usf_acc, usf_thr=usf_thr)
            else:
                usf_err = self.get_mae(
                    dt_usfs,
                    gt_usfs,
                    # (gt_annotated > 0.5).any(axis=1).view(dt_usfs.shape),
                    mask=torch.ones_like(dt_usfs, dtype=torch.bool)
                )
                losses.update(usf_mae=usf_err)

        return losses
    
    def get_pose_accuracy(self, dt, gt, mask):
        """Calculate the accuracy of predicted pose."""
        _, avg_acc, _ = pose_pck_accuracy(
            output=to_numpy(dt),
            target=to_numpy(gt),
            mask=to_numpy(mask),
        )
        acc_pose = torch.tensor(avg_acc, device=gt.device)
        return acc_pose
    
    def get_binary_accuracy(self, dt, gt, mask, force_balanced=False):
        """Calculate the binary accuracy."""
        assert dt.shape == gt.shape
        device = gt.device
        dt = to_numpy(dt)
        gt = to_numpy(gt)
        mask = to_numpy(mask)

        dt = dt[mask]
        gt = gt[mask]
        gt = gt.astype(bool)

        if force_balanced:
            # Force the number of positive and negative samples to be balanced
            pos_num = np.sum(gt)
            neg_num = len(gt) - pos_num
            num = min(pos_num, neg_num)
            if num == 0:
                return torch.tensor([0.0], device=device), torch.tensor([0.0], device=device)
            pos_idx = np.where(gt)[0]
            neg_idx = np.where(~gt)[0]

            # Randomly sample the same number of positive and negative samples
            np.random.shuffle(pos_idx)
            np.random.shuffle(neg_idx)
            idx = np.concatenate([pos_idx[:num], neg_idx[:num]])
            dt = dt[idx]
            gt = gt[idx]

        n_samples = len(gt)
        thresholds = np.arange(0.1, 1.0, 0.05)
        preds = (dt[:, None] > thresholds)
        correct = preds == gt[:, None]
        counts = correct.sum(axis=0)

        # Find the threshold that maximizes the accuracy
        best_idx = np.argmax(counts)
        best_threshold = thresholds[best_idx]
        best_acc = counts[best_idx] / n_samples

        best_acc = torch.tensor(best_acc, device=device).float()
        best_threshold = torch.tensor(best_threshold, device=device).float()
        return best_acc, best_threshold

    def get_mae(self, dt, gt, mask):
        """Calculate the mean absolute error."""
        assert dt.shape == gt.shape
        device = gt.device
        dt = to_numpy(dt)
        gt = to_numpy(gt)
        mask = to_numpy(mask)
        
        dt = dt[mask]
        gt = gt[mask]
        mae = np.abs(dt - gt).mean()

        mae = torch.tensor(mae, device=device)
        return mae

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
                                  **kwargs):
        """A hook function to convert old-version state dict of
        :class:`TopdownHeatmapSimpleHead` (before MMPose v1.0.0) to a
        compatible format of :class:`HeatmapHead`.

        The hook will be automatically registered during initialization.
        """
        version = local_meta.get('version', None)
        if version and version >= self._version:
            return

        # convert old-version state dict
        keys = list(state_dict.keys())
        for _k in keys:
            if not _k.startswith(prefix):
                continue
            v = state_dict.pop(_k)
            k = _k[len(prefix):]
            # In old version, "final_layer" includes both intermediate
            # conv layers (new "conv_layers") and final conv layers (new
            # "final_layer").
            #
            # If there is no intermediate conv layer, old "final_layer" will
            # have keys like "final_layer.xxx", which should be still
            # named "final_layer.xxx";
            #
            # If there are intermediate conv layers, old "final_layer"  will
            # have keys like "final_layer.n.xxx", where the weights of the last
            # one should be renamed "final_layer.xxx", and others should be
            # renamed "conv_layers.n.xxx"
            k_parts = k.split('.')
            if k_parts[0] == 'final_layer':
                if len(k_parts) == 3:
                    assert isinstance(self.conv_layers, nn.Sequential)
                    idx = int(k_parts[1])
                    if idx < len(self.conv_layers):
                        # final_layer.n.xxx -> conv_layers.n.xxx
                        k_new = 'conv_layers.' + '.'.join(k_parts[1:])
                    else:
                        # final_layer.n.xxx -> final_layer.xxx
                        k_new = 'final_layer.' + k_parts[2]
                else:
                    # final_layer.xxx remains final_layer.xxx
                    k_new = k
            else:
                k_new = k

            state_dict[prefix + k_new] = v
