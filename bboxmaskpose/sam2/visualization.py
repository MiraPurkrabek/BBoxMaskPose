import os
import cv2
import numpy as np

from bboxmaskpose.sam2.distinctipy import get_colors

from pycocotools import mask as Mask


def batch_visualize_masks(args, image, masks_rle, image_kpts, bboxes_xyxy, dt_bboxes, gt_masks_raw, bbox_ious, mask_ious, image_path=None, mask_out=False, alpha=1.0):
    # Decode dt_masks_rle
    dt_masks = []
    for mask_rle in masks_rle:
        mask = Mask.decode(mask_rle)
        dt_masks.append(mask) 
    dt_masks = np.array(dt_masks)

    # Decode gt_masks_raw
    gt_masks = []
    for gt_mask in gt_masks_raw:
        if gt_mask is None:
            gt_masks.append(np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8))
        else:
            # gt_mask_rle = Mask.frPyObjects(gt_mask, image.shape[0], image.shape[1])
            # gt_mask_rle = Mask.merge(gt_mask_rle)
            gt_mask_rle = gt_mask
            mask = Mask.decode(gt_mask_rle)
            gt_masks.append(mask)
    gt_masks = np.array(gt_masks)

    # Generate random color for each mask
    if mask_out:
        dt_mask_image = dt_masks.max(axis=0)
        dt_mask_image = (~ dt_mask_image.astype(bool)).astype(np.uint8)
        dt_mask_image = cv2.resize(dt_mask_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        dt_mask_image = image * dt_mask_image[:, :, None]
        dt_mask_image = cv2.addWeighted(image, 1-alpha, dt_mask_image, alpha, 0)
    else:
        colors = (np.array(get_colors(dt_masks.shape[0])) * 255).astype(int)
        
        # colors = np.random.randint(0, 255, (dt_masks.shape[0], 3))
        # # Make sure no colors are too dark
        # np.clip(colors, 50, 255, out=colors)

        # Repeat masks to 3 channels
        dt_masks = np.repeat(dt_masks[:, :, :, None], 3, axis=3)
        gt_masks = np.repeat(gt_masks[:, :, :, None], 3, axis=3)

        # Colorize masks
        dt_masks = dt_masks * colors[:, None, None, :]
        gt_masks = gt_masks * colors[:, None, None, :]

        # # Remove masks that are too small
        # dt_masks_area = dt_masks.any(axis=3).sum(axis=(1, 2))
        # dt_masks[dt_masks_area < 300*300] = 0
            
        # Collapse masks to 3 channels
        dt_mask_image = dt_masks.max(axis=0)
        gt_mask_image = gt_masks.max(axis=0)

        # Convert to uint8
        dt_mask_image = dt_mask_image.astype(np.uint8)
        gt_mask_image = gt_mask_image.astype(np.uint8)

        # Resize masks to image size
        dt_mask_image = cv2.resize(dt_mask_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        gt_mask_image = cv2.resize(gt_mask_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Add masks to image
    if not mask_out:

        dt_mask_image = cv2.addWeighted(image, 0.6, dt_mask_image, 0.4, 0)
        # Draw contours around the masks
        for mask, color in zip(dt_masks, colors):

            color = color.astype(int).tolist()

            mask = mask.astype(np.uint8)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(dt_mask_image, contours, -1, color, 1)

        gt_mask_image = cv2.addWeighted(image, 0.6, gt_mask_image, 0.4, 0)

    # Draw keypoints
    if image_kpts is not None and not mask_out:
        for instance_kpts, color in zip(image_kpts, colors):
            color = tuple(color.astype(int).tolist())
            for kpt in instance_kpts:
                cv2.circle(dt_mask_image, kpt.astype(int)[:2], 3, color, -1)
                cv2.circle(gt_mask_image, kpt.astype(int)[:2], 3, color, -1)

    # Draw bboxes
    if bboxes_xyxy is not None and not mask_out:
        bboxes_xyxy = np.array(bboxes_xyxy)
        dt_bboxes = np.array(dt_bboxes)
        dt_bboxes[:, 2:] += dt_bboxes[:, :2]
        for gt_bbox, dt_bbox, color, biou in zip(bboxes_xyxy, dt_bboxes, colors, bbox_ious):
            color = tuple(color.astype(int).tolist())
            gbox = gt_bbox.astype(int)
            dbox = dt_bbox.astype(int)
            cv2.rectangle(dt_mask_image, (dbox[0], dbox[1]), (dbox[2], dbox[3]), color, 2)
            cv2.rectangle(gt_mask_image, (gbox[0], gbox[1]), (gbox[2], gbox[3]), color, 2)

            # Write IOU on th etop-left corner of the bbox
            # cv2.putText(dt_mask_image, "{:.2f}".format(biou), (dbox[0], dbox[1]-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            # cv2.putText(gt_mask_image, "{:.2f}".format(biou), (gbox[0], gbox[1]-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Save the image
    bbox_ious = np.array(bbox_ious)
    mask_ious = np.array(mask_ious)
    if image_path is not None:
        save_name = os.path.basename(image_path)
    else:
        save_name = "batch_bbox_{:06.2f}_mask_{:06.2f}_{:02d}kpts_{:06d}.jpg".format(
            bbox_ious.mean(), mask_ious.mean(), args.num_pos_keypoints, np.random.randint(1000000),
        )

    if 'debug_folder' not in args:
        args.debug_folder = "debug"

    if mask_out:
        cv2.imwrite(os.path.join(args.debug_folder, save_name), dt_mask_image)               
    else:
        cv2.imwrite(os.path.join(args.debug_folder, save_name), np.hstack([gt_mask_image, dt_mask_image]))      
  
