# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from typing import Optional, Sequence

import numpy as np

import mmcv
import mmengine
import mmengine.fileio as fileio
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer

from mmpose.registry import HOOKS
from mmpose.structures import PoseDataSample, merge_data_samples
from mmpose.structures.keypoint import fix_bbox_aspect_ratio


@HOOKS.register_module()
class PoseVisualizationHook(Hook):
    """Pose Estimation Visualization Hook. Used to visualize validation and
    testing process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.
    2. If ``out_dir`` is specified, it means that the prediction results
        need to be saved to ``out_dir``. In order to avoid vis_backends
        also storing data, so ``vis_backends`` needs to be excluded.
    3. ``vis_backends`` takes effect if the user does not specify ``show``
        and `out_dir``. You can set ``vis_backends`` to WandbVisBackend or
        TensorboardVisBackend to store the prediction result in Wandb or
        Tensorboard.

    Args:
        enable (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        score_thr (float): The threshold to visualize the bboxes
            and masks. Defaults to 0.3.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        out_dir (str, optional): directory where painted images
            will be saved in testing process.
        backend_args (dict, optional): Arguments to instantiate the preifx of
            uri corresponding backend. Defaults to None.
    """

    def __init__(
        self,
        enable: bool = False,
        interval: int = 50,
        kpt_thr: float = 0.3,
        show: bool = False,
        wait_time: float = 0.,
        out_dir: Optional[str] = None,
        backend_args: Optional[dict] = None,
    ):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.kpt_thr = kpt_thr
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.enable = enable
        self.out_dir = out_dir
        self._test_index = 0
        self.backend_args = backend_args

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[PoseDataSample]) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`PoseDataSample`]): Outputs from model.
        """
        if self.enable is False:
            return

        self._visualizer.set_dataset_meta(runner.val_evaluator.dataset_meta)

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx

        # Visualize only the first data
        img_path = data_batch['data_samples'][0].get('img_path')
        img_bytes = fileio.get(img_path, backend_args=self.backend_args)
        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
        data_sample = outputs[0]

        # revert the heatmap on the original image
        data_sample = merge_data_samples([data_sample])

        if total_curr_iter % self.interval == 0:
            self._visualizer.add_datasample(
                os.path.basename(img_path) if self.show else 'val_img',
                img,
                data_sample=data_sample,
                draw_gt=False,
                draw_bbox=True,
                draw_heatmap=True,
                show=self.show,
                wait_time=self.wait_time,
                kpt_thr=self.kpt_thr,
                step=total_curr_iter)

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[PoseDataSample]) -> None:
        """Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`PoseDataSample`]): Outputs from model.
        """
        if self.enable is False:
            return

        if self.out_dir is not None:
            self.out_dir = os.path.join(runner.work_dir, runner.timestamp,
                                        self.out_dir)
            mmengine.mkdir_or_exist(self.out_dir)

        self._visualizer.set_dataset_meta(runner.test_evaluator.dataset_meta)

        for data_sample in outputs:
            self._test_index += 1

            img_path = data_sample.get('img_path')
            img_bytes = fileio.get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')

            # img = pad_img_to_amap(img, data_sample)
            
            data_sample = merge_data_samples([data_sample])

            # Resize image to heatmap size
            if data_sample.get('_pred_heatmaps') is not None:
                heatmap_size = data_sample._pred_heatmaps.shape
                img = mmcv.imresize(img, heatmap_size[::-1])

            out_file = None
            if self.out_dir is not None:
                out_file_name, postfix = os.path.basename(img_path).rsplit(
                    '.', 1)
                index = len([
                    fname for fname in os.listdir(self.out_dir)
                    if fname.startswith(out_file_name)
                ])
                out_file = f'{out_file_name}_{index}.{postfix}'
                out_file = os.path.join(self.out_dir, out_file)

            self._visualizer.add_datasample(
                os.path.basename(img_path) if self.show else 'test_img',
                img,
                data_sample=data_sample,
                show=self.show,
                draw_gt=False,
                draw_bbox=True,
                draw_heatmap=True,
                wait_time=self.wait_time,
                kpt_thr=self.kpt_thr,
                out_file=out_file,
                step=self._test_index)


def pad_img_to_amap(img, data_sample):
    bbox_xywh = None
    if 'raw_ann_info' in data_sample:
        bbox_xywh = data_sample.raw_ann_info['bbox']
    elif 'pred_instances' in data_sample:
        bbox_xywh = data_sample.pred_instances.bboxes.flatten()
    
    if bbox_xywh is None:
        return img

    bbox_xyxy = np.array([
        bbox_xywh[0], bbox_xywh[1],
        bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]
    ])
    abox_xyxy = fix_bbox_aspect_ratio(bbox_xyxy, aspect_ratio=3/4, padding=1.25, bbox_format='xyxy')
    abox_xyxy = abox_xyxy.flatten()

    x_pad = np.array([max(0, -abox_xyxy[0]), max(0, abox_xyxy[2] - img.shape[1])], dtype=int)
    y_pad = np.array([max(0, -abox_xyxy[1]), max(0, abox_xyxy[3] - img.shape[0])], dtype=int)
    img = np.pad(img, ((y_pad[0], y_pad[1]), (x_pad[0], x_pad[1]), (0, 0)), mode='constant')

    kpts = data_sample.pred_instances.keypoints[0].reshape(-1, 2)
    kpts[:, :2] += np.array([x_pad[0], y_pad[0]])
    data_sample.pred_instances.keypoints[0] = kpts.reshape(data_sample.pred_instances.keypoints[0].shape)

    return img
