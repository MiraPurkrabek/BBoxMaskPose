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

from sparsemax import Sparsemax

import os
import shutil
import cv2

from mmpose.structures.keypoint import fix_bbox_aspect_ratio

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class CalibrationHead(BaseHead):
    """Multi-variate head predicting all information about keypoints. Apart 
    from the heatmap, it also predicts:
        1) Heatmap for each keypoint
        2) Probability of keypoint being in the heatmap
        3) Visibility of each keypoint
        4) Predicted OKS per keypoint
        5) Predictd euclidean error per keypoint
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
        probability_loss (Config): Config of the probability loss. Defaults to use
            :class:`BCELoss`
        visibility_loss (Config): Config of the visibility loss. Defaults to use
            :class:`BCELoss`
        oks_loss (Config): Config of the oks loss. Defaults to use
            :class:`MSELoss`
        error_loss (Config): Config of the error loss. Defaults to use
            :class:`L1LogLoss`
        normalize (bool): Whether to normalize values in the heatmaps between 
            0 and 1 with sigmoid. Defaults to ``False``
        detach_probability (bool): Whether to detach the probability
            from gradient computation. Defaults to ``True``
        detach_visibility (bool): Whether to detach the visibility
            from gradient computation. Defaults to ``True``
        learn_heatmaps_from_zeros (bool): Whether to learn the
            heatmaps from zeros. Defaults to ``False``
        freeze_heatmaps (bool): Whether to freeze the heatmaps prediction.
            Defaults to ``False``
        freeze_probability (bool): Whether to freeze the probability prediction.
            Defaults to ``False``
        freeze_visibility (bool): Whether to freeze the visibility prediction.
            Defaults to ``False``
        freeze_oks (bool): Whether to freeze the oks prediction.
            Defaults to ``False``
        freeze_error (bool): Whether to freeze the error prediction.
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
                 probability_loss: ConfigType = dict(
                     type='BCELoss', use_target_weight=True),
                 visibility_loss: ConfigType = dict(
                     type='BCELoss', use_target_weight=True),
                 oks_loss: ConfigType = dict(
                     type='MSELoss', use_target_weight=True),
                 error_loss: ConfigType = dict(
                     type='L1LogLoss', use_target_weight=True),
                 normalize: float = None,
                 detach_probability: bool = True,
                 detach_visibility: bool = True,
                 learn_heatmaps_from_zeros: bool = False,
                 freeze_heatmaps: bool = False,
                 freeze_probability: bool = False,
                 freeze_visibility: bool = False,
                 freeze_oks: bool = False,
                 freeze_error: bool = False,
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
        self.probability_loss_module = MODELS.build(probability_loss)
        self.visibility_loss_module = MODELS.build(visibility_loss)
        self.oks_loss_module = MODELS.build(oks_loss)
        self.error_loss_module = MODELS.build(error_loss)

        self.temperature = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)

        self.gauss_sigma = 2.0
        self.gauss_kernel_size = int(2.0 * 3.0 * self.gauss_sigma + 1.0)
        ts = torch.linspace(
            - self.gauss_kernel_size // 2,
            self.gauss_kernel_size // 2,
            self.gauss_kernel_size
        )
        gauss = torch.exp(-(ts / self.gauss_sigma)**2 / 2)
        gauss = gauss / gauss.sum()
        self.gauss_kernel = gauss.unsqueeze(0) * gauss.unsqueeze(1)

        self.decoder = KEYPOINT_CODECS.build(decoder)
        self.nonlinearity = nn.ReLU(inplace=True)
        self.learn_heatmaps_from_zeros = learn_heatmaps_from_zeros

        self.num_iters = 0
        unique_hash = np.random.randint(0, 100000)
        self.loss_vis_folder = "work_dirs/loss_vis_{:05d}".format(unique_hash)
        self.interval = 50
        shutil.rmtree(self.loss_vis_folder, ignore_errors=True)
        
        self._build_heatmap_head(
            in_channels=in_channels,
            out_channels=out_channels,
            deconv_out_channels=deconv_out_channels,
            deconv_kernel_sizes=deconv_kernel_sizes,
            conv_out_channels=conv_out_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            final_layer_dict=final_layer_dict,
            normalize=normalize,
            freeze=freeze_heatmaps)
        
        self.normalize = normalize
        
        self.detach_probability = detach_probability
        self._build_probability_head(
            in_channels=in_channels,
            out_channels=out_channels,
            freeze=freeze_probability)
        
        self.detach_visibility = detach_visibility
        self._build_visibility_head(
            in_channels=in_channels,
            out_channels=out_channels,
            freeze=freeze_visibility)
        
        self._build_oks_head(
            in_channels=in_channels,
            out_channels=out_channels,
            freeze=freeze_oks)
        self.freeze_oks = freeze_oks

        self._build_error_head(
            in_channels=in_channels,
            out_channels=out_channels,
            freeze=freeze_error)
        self.freeze_error = freeze_error

        # Register the hook to automatically convert old version state dicts
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

        self._freeze_all_but_temperature()

        # Print all params and their gradients
        print("\n", "="*20)
        for name, param in self.named_parameters():
            print(name, param.requires_grad)


    def _freeze_all_but_temperature(self):
        for param in self.parameters():
            param.requires_grad = False
        self.temperature.requires_grad = True

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
        # self.normalize_layer = lambda x: x / x.sum(dim=-1, keepdim=True) if normalize else nn.Identity()
        # self.normalize_layer = nn.Softmax(dim=-1) if normalize else nn.Identity()
        self.normalize_layer = nn.Identity() if normalize is None else Sparsemax(dim=-1)

        if freeze:
            for param in self.deconv_layers.parameters():
                param.requires_grad = False
            for param in self.conv_layers.parameters():
                param.requires_grad = False
            for param in self.final_layer.parameters():
                param.requires_grad = False

    def _build_probability_head(self, in_channels: int, out_channels: int,
                                freeze: bool = False) -> nn.Module:
        """Build the probability head module."""
        ppb_layers = []
        kernel_sizes = [(4, 3), (2, 2), (2, 2)]
        for i in range(len(kernel_sizes)):
            ppb_layers.append(
                build_conv_layer(
                    dict(type='Conv2d'),
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            ppb_layers.append(
                nn.BatchNorm2d(num_features=in_channels))
            ppb_layers.append(
                nn.MaxPool2d(kernel_size=kernel_sizes[i], stride=kernel_sizes[i], padding=0))
            ppb_layers.append(self.nonlinearity)
        ppb_layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0))
        ppb_layers.append(nn.Sigmoid())
        self.probability_layers = nn.Sequential(*ppb_layers)

        if freeze:
            for param in self.probability_layers.parameters():
                param.requires_grad = False

    def _build_visibility_head(self, in_channels: int, out_channels: int,
                                 freeze: bool = False) -> nn.Module:
        """Build the visibility head module."""
        vis_layers = []
        kernel_sizes = [(4, 3), (2, 2), (2, 2)]
        for i in range(len(kernel_sizes)):
            vis_layers.append(
                build_conv_layer(
                    dict(type='Conv2d'),
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            vis_layers.append(
                nn.BatchNorm2d(num_features=in_channels))
            vis_layers.append(
                nn.MaxPool2d(kernel_size=kernel_sizes[i], stride=kernel_sizes[i], padding=0))
            vis_layers.append(self.nonlinearity)
        vis_layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0))
        vis_layers.append(nn.Sigmoid())
        self.visibility_layers = nn.Sequential(*vis_layers)

        if freeze:
            for param in self.visibility_layers.parameters():
                param.requires_grad = False

    def _build_oks_head(self, in_channels: int, out_channels: int,
                        freeze: bool = False) -> nn.Module:
        """Build the oks head module."""
        oks_layers = []
        kernel_sizes = [(4, 3), (2, 2), (2, 2)]
        for i in range(len(kernel_sizes)):
            oks_layers.append(
                build_conv_layer(
                    dict(type='Conv2d'),
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            oks_layers.append(
                nn.BatchNorm2d(num_features=in_channels))
            oks_layers.append(
                nn.MaxPool2d(kernel_size=kernel_sizes[i], stride=kernel_sizes[i], padding=0))
            oks_layers.append(self.nonlinearity)
        oks_layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0))
        oks_layers.append(nn.Sigmoid())
        self.oks_layers = nn.Sequential(*oks_layers)

        if freeze:
            for param in self.oks_layers.parameters():
                param.requires_grad = False

    def _build_error_head(self, in_channels: int, out_channels: int,
                        freeze: bool = False) -> nn.Module:
        """Build the error head module."""
        error_layers = []
        kernel_sizes = [(4, 3), (2, 2), (2, 2)]
        for i in range(len(kernel_sizes)):
            error_layers.append(
                build_conv_layer(
                    dict(type='Conv2d'),
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            error_layers.append(
                nn.BatchNorm2d(num_features=in_channels))
            error_layers.append(
                nn.MaxPool2d(kernel_size=kernel_sizes[i], stride=kernel_sizes[i], padding=0))
            error_layers.append(self.nonlinearity)
        error_layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0))
        error_layers.append(self.nonlinearity)
        self.error_layers = nn.Sequential(*error_layers)

        if freeze:
            for param in self.error_layers.parameters():
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

    def _error_from_heatmaps(self, gt_heatmaps: Tensor, dt_heatmaps: Tensor) -> Tensor:
        """Calculate the error from heatmaps.

        Args:
            heatmaps (Tensor): The predicted heatmaps.

        Returns:
            Tensor: The predicted error.
        """
        # Transform to numpy
        gt_heatmaps = to_numpy(gt_heatmaps)
        dt_heatmaps = to_numpy(dt_heatmaps)

        # Get locations from heatmaps
        B, C, H, W = gt_heatmaps.shape
        gt_coords = np.zeros((B, C, 2))
        dt_coords = np.zeros((B, C, 2))
        for i, (gt_htm, dt_htm) in enumerate(zip(gt_heatmaps, dt_heatmaps)):
            coords, score = self.decoder.decode(gt_htm)
            coords = coords.squeeze()
            gt_coords[i, :, :] = coords
            
            coords, score = self.decoder.decode(dt_htm)
            coords = coords.squeeze()
            dt_coords[i, :, :] = coords
        
        # NaN coordinates mean empty heatmaps -> set them to -1
        # as the error will be ignored by weight
        gt_coords[np.isnan(gt_coords)] = -1

        # Calculate the error
        target_errors = np.linalg.norm(gt_coords - dt_coords, axis=2)
        assert (target_errors >= 0).all(), "Euclidean distance cannot be negative"

        return target_errors
    
    def _oks_from_heatmaps(self, gt_heatmaps: Tensor, dt_heatmaps: Tensor, weight: Tensor) -> Tensor:
        """Calculate the OKS from heatmaps.

        Args:
            heatmaps (Tensor): The predicted heatmaps.

        Returns:
            Tensor: The predicted OKS.
        """
        C = dt_heatmaps.shape[1]

        # Transform to numpy
        gt_heatmaps = to_numpy(gt_heatmaps)
        dt_heatmaps = to_numpy(dt_heatmaps)
        B, C, H, W = gt_heatmaps.shape
        weight = to_numpy(weight).squeeze().reshape((B, C, 1))

        # Get locations from heatmaps
        gt_coords = np.zeros((B, C, 2))
        dt_coords = np.zeros((B, C, 2))
        for i, (gt_htm, dt_htm) in enumerate(zip(gt_heatmaps, dt_heatmaps)):
            coords, score = self.decoder.decode(gt_htm)
            coords = coords.squeeze()
            gt_coords[i, :, :] = coords
            
            coords, score = self.decoder.decode(dt_htm)
            coords = coords.squeeze()
            dt_coords[i, :, :] = coords

        # NaN coordinates mean empty heatmaps -> set them to 0
        gt_coords[np.isnan(gt_coords)] = 0

        # Add probability as visibility
        gt_coords = gt_coords * weight
        dt_coords = dt_coords * weight
        gt_coords = np.concatenate((gt_coords, weight*2), axis=2)
        dt_coords = np.concatenate((dt_coords, weight*2), axis=2)

        # Calculate the oks
        target_oks = []
        oks_weights = []
        for i in range(len(gt_coords)):
            gt_kpts = gt_coords[i]
            dt_kpts = dt_coords[i]
            valid_gt_kpts = gt_kpts[:, 2] > 0
            if not valid_gt_kpts.any():
                # Changed for per-keypoint OKS
                target_oks.append(np.zeros(C))
                oks_weights.append(0)
                continue

            gt_bbox = np.array([
                0, 0,
                64, 48,
            ])
            gt = {
                'keypoints': gt_kpts,
                'bbox': gt_bbox,
                'area': gt_bbox[2] * gt_bbox[3],
            }
            dt = {
                'keypoints': dt_kpts,
                'bbox': gt_bbox,
                'area': gt_bbox[2] * gt_bbox[3],
            }
            # Changed for per-keypoint OKS
            oks = compute_oks(gt, dt, use_area=False, per_kpt=True)
            target_oks.append(oks)
            oks_weights.append(1)

        target_oks = np.array(target_oks)
        target_oks = torch.from_numpy(target_oks).float()

        oks_weights = np.array(oks_weights)
        oks_weights = torch.from_numpy(oks_weights).float()

        return target_oks, oks_weights

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(
                type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1)
        ]
        return init_cfg

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Forward the network. The input is multi scale feature maps and the
        output is (1) the heatmap, (2) probability, (3) visibility, (4) oks and (5) error.

        Args:
            feats (Tensor): Multi scale feature maps.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: outputs.
        """
        x = feats[-1].detach()

        heatmaps = self.forward_heatmap(x)
        probabilities = self.forward_probability(x)
        visibilities = self.forward_visibility(x)
        oks = self.forward_oks(x)
        errors = self.forward_error(x)

        return heatmaps, probabilities, visibilities, oks, errors
    
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
        B, C, H, W = x.shape
        x = x.reshape((B, C, H*W))
        x = self.normalize_layer(x/self.temperature)
        if self.normalize is not None:
            x = x * self.normalize
        x = torch.clamp(x, 0, 1)
        x = x.reshape((B, C, H, W))

        # # Blur the heatmaps with Gaussian
        # x = x.reshape((B*C, 1, H, W))
        # x = nn.functional.conv2d(x, self.gauss_kernel[None, None, :, :].to(x.device), padding='same')
        # x = x.reshape((B, C, H, W))

        return x
    
    def forward_probability(self, x: Tensor) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the probability.

        Args:
            x (Tensor): Multi scale feature maps.
            detach (bool): Whether to detach the probability from gradient

        Returns:
            Tensor: output probability.
        """
        if self.detach_probability:
            x = x.detach()
        x = self.probability_layers(x)
        return x

    def forward_visibility(self, x: Tensor) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the visibility.

        Args:
            x (Tensor): Multi scale feature maps.
            detach (bool): Whether to detach the visibility from gradient

        Returns:
            Tensor: output visibility.
        """
        if self.detach_visibility:
            x = x.detach()
        x = self.visibility_layers(x)
        return x
    
    def forward_oks(self, x: Tensor) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the oks.

        Args:
            x (Tensor): Multi scale feature maps.

        Returns:
            Tensor: output oks.
        """
        x = x.detach()
        x = self.oks_layers(x)
        return x
    
    def forward_error(self, x: Tensor) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the euclidean error.

        Args:
            x (Tensor): Multi scale feature maps.

        Returns:
            Tensor: output error.
        """
        x = x.detach()
        x = self.error_layers(x)
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

            _htm, _prob, _vis, _oks, _err = self.forward(_feats)
            _htm_flip, _prob_flip, _vis_flip, _oks_flip, _err_flip = self.forward(_feats_flip)
            B, C, H, W = _htm.shape

            # Flip back the keypoints
            _htm_flip = flip_heatmaps(
                _htm_flip,
                flip_mode=test_cfg.get('flip_mode', 'heatmap'),
                flip_indices=flip_indices,
                shift_heatmap=test_cfg.get('shift_heatmap', False))
            heatmaps = (_htm + _htm_flip) * 0.5

            # Flip back scalars
            _prob_flip = _prob_flip[:, flip_indices]
            _vis_flip = _vis_flip[:, flip_indices]
            _oks_flip = _oks_flip[:, flip_indices]
            _err_flip = _err_flip[:, flip_indices]
            
            probabilities = (_prob + _prob_flip) * 0.5
            visibilities = (_vis + _vis_flip) * 0.5
            oks = (_oks + _oks_flip) * 0.5
            errors = (_err + _err_flip) * 0.5
        else:
            heatmaps, probabilities, visibilities, oks, errors = self.forward(feats)
            B, C, H, W = heatmaps.shape

        preds = self.decode(heatmaps)
        probabilities = to_numpy(probabilities).reshape((B, 1, C))
        visibilities = to_numpy(visibilities).reshape((B, 1, C))
        oks = to_numpy(oks).reshape((B, 1, C))
        errors = to_numpy(errors).reshape((B, 1, C))
        
        # Normalize errors by dividing with the diagonal of the heatmap
        htm_diagonal = np.sqrt(H**2 + W**2)
        errors = errors / htm_diagonal

        for pi, p in enumerate(preds):
            p.set_field(p['keypoint_scores'], "keypoints_conf")
            p.set_field(probabilities[pi], "keypoints_probs")
            p.set_field(visibilities[pi], "keypoints_visible")
            p.set_field(oks[pi], "keypoints_oks")
            p.set_field(errors[pi], "keypoints_error")

            # Replace the keypoint scores with OKS/errors
            if not self.freeze_oks:
                p.set_field(oks[pi], "keypoint_scores")
            # p.set_field(1-errors[pi], "keypoint_scores")

        # hm = heatmaps.detach().cpu().numpy()
        # print("Heatmaps:", hm.shape, hm.min(), hm.max())
            
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
        dt_heatmaps, dt_probs, dt_vis, dt_oks, dt_errs = self.forward(feats)
        device=dt_heatmaps.device
        B, C, H, W = dt_heatmaps.shape
        
        # Extract GT data
        gt_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in batch_data_samples])
        gt_probs = np.stack(
            [d.gt_instances.in_image.astype(int) for d in batch_data_samples])
        gt_annotated = np.stack(
            [d.gt_instances.keypoints_visible.astype(int) for d in batch_data_samples])
        gt_vis = np.stack(
            [d.gt_instances.keypoints_visibility.astype(int) for d in batch_data_samples])
        keypoint_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])

        # Compute GT errors and OKS
        if self.freeze_error:
            gt_errs = torch.zeros((B, C, 1), device=device, dtype=dt_errs.dtype)
        else:
            gt_errs = self._error_from_heatmaps(gt_heatmaps, dt_heatmaps)
        if self.freeze_oks:
            gt_oks = torch.zeros((B, C, 1), device=device, dtype=dt_oks.dtype)
            oks_weight = torch.zeros((B, C, 1), device=device, dtype=dt_oks.dtype)
        else:
            gt_oks, oks_weight = self._oks_from_heatmaps(
                gt_heatmaps,
                dt_heatmaps,
                gt_probs & gt_annotated,
            )

        # Convert everything to tensors
        gt_probs = torch.tensor(gt_probs, device=device, dtype=dt_probs.dtype)
        gt_vis = torch.tensor(gt_vis, device=device, dtype=dt_vis.dtype)
        gt_annotated = torch.tensor(gt_annotated, device=device)
        
        gt_oks = gt_oks.to(device).to(dt_oks.dtype)
        oks_weight = oks_weight.to(device).to(dt_oks.dtype)
        gt_errs = gt_errs.to(device).to(dt_errs.dtype)

        # Reshape everything to comparable shapes
        gt_heatmaps = gt_heatmaps.view((B, C, H, W))
        dt_heatmaps = dt_heatmaps.view((B, C, H, W))
        gt_probs = gt_probs.view((B, C))
        dt_probs = dt_probs.view((B, C))
        gt_vis = gt_vis.view((B, C))
        dt_vis = dt_vis.view((B, C))
        gt_oks = gt_oks.view((B, C))
        dt_oks = dt_oks.view((B, C))
        gt_errs = gt_errs.view((B, C))
        dt_errs = dt_errs.view((B, C))
        keypoint_weights = keypoint_weights.view((B, C))
        gt_annotated = gt_annotated.view((B, C))
        # oks_weight = oks_weight.view((B, C))

        annotated_in = gt_annotated & (gt_probs > 0.5)

        # calculate losses
        losses = dict()
        if self.learn_heatmaps_from_zeros:
            heatmap_weights = gt_annotated
        else:
            # heatmap_weights = keypoint_weights
            heatmap_weights = annotated_in

        heatmap_loss_pxl = self.keypoint_loss_module(dt_heatmaps, gt_heatmaps, annotated_in, per_pixel=True)
        heatmap_loss     = self.keypoint_loss_module(dt_heatmaps, gt_heatmaps, annotated_in)
        # probability_loss = self.probability_loss_module(dt_probs, gt_probs, gt_annotated)
        # visibility_loss  = self.visibility_loss_module(dt_vis, gt_vis, annotated_in)
        # oks_loss         = self.oks_loss_module(dt_oks, gt_oks, annotated_in)
        # error_loss       = self.error_loss_module(dt_errs, gt_errs, annotated_in)    

        losses.update(
            loss_kpt=heatmap_loss
        )
        
        # calculate accuracy
        if train_cfg.get('compute_acc', True):
            acc_pose = self.get_pose_accuracy(
                dt_heatmaps, gt_heatmaps, keypoint_weights > 0.5
            )
            losses.update(acc_pose=acc_pose)

            # Calculate the best binary accuracy for probability
            acc_prob, _ = self.get_binary_accuracy(
                dt_probs,
                gt_probs,
                gt_annotated > 0.5,
                force_balanced=True,
            )
            losses.update(acc_prob=acc_prob)

            # Calculate the best binary accuracy for visibility
            acc_vis, _ = self.get_binary_accuracy(
                dt_vis,
                gt_vis,
                annotated_in > 0.5,
                force_balanced=True,
            )
            losses.update(acc_vis=acc_vis)

            # Calculate the MAE for OKS
            acc_oks = self.get_mae(
                dt_oks,
                gt_oks,
                annotated_in > 0.5,
            )
            losses.update(mae_oks=acc_oks)

            # Calculate the MAE for euclidean error
            acc_err = self.get_mae(
                dt_errs,
                gt_errs,
                annotated_in > 0.5,
            )
            losses.update(mae_err=acc_err)

            # Calculate the MAE between Euclidean error and OKS
            err_to_oks_mae = self.get_mae(
                self.error_to_OKS(dt_errs, area=H*W),
                gt_oks,
                annotated_in > 0.5,
            )
            losses.update(mae_err_to_oks=err_to_oks_mae)

        print(self.temperature.item())

        return losses
    
    def _visualize_heatmaps(
        self,
        htm,
        tgt,
        lss,
        weight,
        prob
    ):
        tgt_range = (tgt.min(), tgt.max())
        htm_range = (htm.min(), htm.max())
        lss_range = (lss.min(), lss.max())

        tgt[tgt < 0] = 0
        htm[htm < 0] = 0
        lss[lss < 0] = 0
        
        # Normalize heatmaps between 0 and 1
        tgt /= (tgt.max()+1e-10)
        htm /= (htm.max()+1e-10)
        lss /= (lss.max()+1e-10)

        scale = 6
        
        htm_color = cv2.cvtColor((htm*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        htm_color = cv2.applyColorMap(htm_color, cv2.COLORMAP_JET)
        htm_color = cv2.resize(htm_color, (htm.shape[1]*scale, htm.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
        
        tgt_color = cv2.cvtColor((tgt*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        tgt_color = cv2.applyColorMap(tgt_color, cv2.COLORMAP_JET)
        tgt_color = cv2.resize(tgt_color, (htm.shape[1]*scale, htm.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
        
        lss_color = cv2.cvtColor((lss*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        lss_color = cv2.applyColorMap(lss_color, cv2.COLORMAP_JET)
        lss_color = cv2.resize(lss_color, (htm.shape[1]*scale, htm.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
        
        if scale > 2:
            tgt_color_text = tgt_color.copy()
            cv2.putText(tgt_color_text, "tgt ({:.1f}, {:.1f})".format(tgt_range[0]*10, tgt_range[1]*10), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tgt_color = cv2.addWeighted(tgt_color, 0.6, tgt_color_text, 0.4, 0)
            
            htm_color_text = htm_color.copy()
            cv2.putText(htm_color_text, "htm ({:.1f}, {:.1f})".format(htm_range[0]*10, htm_range[1]*10), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            htm_color = cv2.addWeighted(htm_color, 0.6, htm_color_text, 0.4, 0)

            lss_color_text = lss_color.copy()
            cv2.putText(lss_color_text, "lss ({:.1f}, {:.1f})".format(lss_range[0]*10, lss_range[1]*10), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            lss_color = cv2.addWeighted(lss_color, 0.6, lss_color_text, 0.4, 0)

            # Get argmax of the target and draw horizontal and vertical lines
            tgt_argmax = np.unravel_index(tgt.argmax(), tgt.shape)
            tgt_color_line = tgt_color.copy()
            cv2.line(tgt_color_line, (0, tgt_argmax[0]*scale), (tgt_color.shape[1], tgt_argmax[0]*scale), (0, 255, 255), 1)
            cv2.line(tgt_color_line, (tgt_argmax[1]*scale, 0), (tgt_argmax[1]*scale, tgt_color.shape[0]), (0, 255, 255), 1)
            tgt_color = cv2.addWeighted(tgt_color, 0.6, tgt_color_line, 0.4, 0)
            htm_color_line = htm_color.copy()
            cv2.line(htm_color_line, (0, tgt_argmax[0]*scale), (tgt_color.shape[1], tgt_argmax[0]*scale), (0, 255, 255), 1)
            cv2.line(htm_color_line, (tgt_argmax[1]*scale, 0), (tgt_argmax[1]*scale, tgt_color.shape[0]), (0, 255, 255), 1)
            htm_color = cv2.addWeighted(htm_color, 0.6, htm_color_line, 0.4, 0)
            lss_color_line = lss_color.copy()
            cv2.line(lss_color_line, (0, tgt_argmax[0]*scale), (tgt_color.shape[1], tgt_argmax[0]*scale), (0, 255, 255), 1)
            cv2.line(lss_color_line, (tgt_argmax[1]*scale, 0), (tgt_argmax[1]*scale, tgt_color.shape[0]), (0, 255, 255), 1)
            lss_color = cv2.addWeighted(lss_color, 0.6, lss_color_line, 0.4, 0)

        white_column = np.ones((tgt_color.shape[0], 1, 3), dtype=np.uint8) * 255

        save_img = np.concatenate((
            tgt_color,
            white_column,
            htm_color,
            white_column,
            lss_color,
        ), axis=1)
        
        if weight < 0.5:
            # Draw a red X across the whole save_img
            cv2.line(save_img, (0, 0), (save_img.shape[1], save_img.shape[0]), (0, 0, 255), 2)
            cv2.line(save_img, (0, save_img.shape[0]), (save_img.shape[1], 0), (0, 0, 255), 2)
        elif prob < 0.5:
            # Draw an yellow X across the whole save_img
            cv2.line(save_img, (0, 0), (save_img.shape[1], save_img.shape[0]), (0, 255, 255), 2)
            cv2.line(save_img, (0, save_img.shape[0]), (save_img.shape[1], 0), (0, 255, 255), 2)
        return save_img

    
    def get_pose_accuracy(self, dt, gt, mask):
        """Calculate the accuracy of predicted pose."""
        _, avg_acc, _ = pose_pck_accuracy(
            output=to_numpy(dt),
            target=to_numpy(gt),
            mask=to_numpy(mask),
            method='argmax',
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

    def error_to_OKS(self, error, area=1.0):
        """Convert the error to OKS."""
        sigmas = np.array(
                [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89])/10.0
        if isinstance(error, torch.Tensor):
            sigmas = torch.tensor(sigmas, device=error.device)
        vars = (sigmas * 2)**2
        norm_error = error**2 / vars / area / 2.0
        return torch.exp(-norm_error)


def compute_oks(gt, dt, use_area=True, per_kpt=False):
    sigmas = np.array(
                [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89])/10.0
    vars = (sigmas * 2)**2
    k = len(sigmas)
    visibility_condition = lambda x: x > 0
    g = np.array(gt['keypoints']).reshape(k, 3)
    xg = g[:, 0]; yg = g[:, 1]; vg = g[:, 2]
    k1 = np.count_nonzero(visibility_condition(vg))
    bb = gt['bbox']
    x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
    y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
    
    d = np.array(dt['keypoints']).reshape((k, 3))
    xd = d[:, 0]; yd = d[:, 1]
            
    if k1>0:
        # measure the per-keypoint distance if keypoints visible
        dx = xd - xg
        dy = yd - yg

    else:
        # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
        z = np.zeros((k))
        dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
        dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)

    if use_area:
        e = (dx**2 + dy**2) / vars / (gt['area']+np.spacing(1)) / 2
    else:
        tmparea = gt['bbox'][3] * gt['bbox'][2] * 0.53
        e = (dx**2 + dy**2) / vars / (tmparea+np.spacing(1)) / 2
        
    if per_kpt:
        oks = np.exp(-e)
        if k1 > 0:
            oks[~visibility_condition(vg)] = 0

    else:
        if k1 > 0:
            e=e[visibility_condition(vg)]
        oks = np.sum(np.exp(-e)) / e.shape[0]

    return oks