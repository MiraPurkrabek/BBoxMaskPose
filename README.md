<!-- omit in toc -->
# Detection, Pose Estimation and Segmentation for Multiple Bodies: Closing the Virtuous Circle

The official repository introducing MaskPose and BBox-Mask-Pose methods.

<h4 align="center">
  <a href="https://mirapurkrabek.github.io/BBox-Mask-Pose/">Project webpage</a> |
  <a href="https://arxiv.org/abs/2307.06737">ArXiv</a> | 
  <a href="https://youtu.be/U05yUP4b2LQ">Video</a>

  <br/>
  <br/>
  <img src="images/004806_BMP.gif" alt="BBox-Mask-Pose loop">
</h4>

<!-- omit in toc -->
## Table of Contents
- [Description](#description)
- [News](#news)
- [Roadmap](#roadmap)
- [Installation](#installation)
- [Results \& Weights](#results--weights)
  - [Pose](#pose)
  - [Detection](#detection)
- [Licence](#licence)
- [Acknowledgements](#acknowledgements)
- [Citation and Contact](#citation-and-contact)


## Description

## News

- 23 Nov 2024: The [project website](https://MiraPurkrabek.github.io/BBox-Mask-Pose) is on

## Roadmap

- [x] Update README with full explanation
- [ ] Upload MaskPose code
- [ ] Upload MaskPose weights
- [ ] Upload RTMDet weights
- [ ] Delete this roadmap

## Installation
  
The code builds on [MMPose](https://github.com/open-mmlab/mmpose). If you want only inference, you can install it simply with:

```console
pip install -U openmim
mim install mmpose
```

And then install our code using 

```
pip install -e .
```

## Results & Weights

### Pose

Results on COCO val and OCHuman val of different Human Pose Estimation (HPE) methods. All results are with detection from [RTMDet-l](https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet) from MMDetection.

We provide trained weights for both MaskPose (introduced in the paper) and ViTPose trained in the multi-dataset setup. Multi-dataset ViTPose is not new but their weights are not compatible with popular MMPose 2.0 codebase. We retrained theme in the MMPose 2.0 environment. 

| Model      | Datasets      | COCO AP | OCHuman AP | weights | notes                                             |
| ---------- | ------------- | ------- | ---------- | ------- | ------------------------------------------------- |
| ViTPose-b  | COCO+AIC+MPII | 76.3    | 42.5       | LINK    | multi-dataset training compatible with MMPose 2.0 |
| MaskPose-b | COCO+AIC+MPII | 76.4    | 45.3       | LINK    |                                                   |

### Detection

To run BBox-Mask-Pose loop, you also need to adapt a detector. We fine-tuned RTMDet-l with mask-out data augmentation as shown in the paper. Weights of the fine-tuned model (compatible with MMDetection config) is LINK. 


## Licence

Please read carefully the [terms and conditions](./LICENSE) and any accompanying documentation before you download and/or use the RePoGen model, data and software, (the "Model & Software"). By downloading and/or using the Model & Software (including downloading, cloning, installing, and any other use of this github repository), you acknowledge that you have read these terms and conditions, understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you must not download and/or use the Model & Software. Any infringement of the terms of this agreement will automatically terminate your rights under this [License](./LICENSE).

## Acknowledgements

The code combines [MMDetection](https://github.com/open-mmlab/mmdetection), [MMPose 2.0](https://github.com/open-mmlab/mmpose) and [ViTPose](https://github.com/ViTAE-Transformer/ViTPose).

## Citation and Contact

The code was implemented by [Miroslav Purkrábek]([htt]https://mirapurkrabek.github.io/).

For questions, please use the Issues of Discussion.

```
BibTeX will be here once submitted to ArXiv 
``````