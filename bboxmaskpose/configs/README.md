# Configuration Files Overview

This directory contains configuration files for reproducing experiments and running inference across different components of the BBoxMaskPose project.

## Which configs are available?

Here you can find configs setting-up hyperparameters of the whole loop.
These are mainly:
- How to prompt SAM
- Which models to use (detection, pose, SAM)
- How to chain models
- ...

For easier reference, the configs have the same names as in the supplementary material of the ICCV paper.
So for example config [**bmp_D3.yaml**](bmp_D3.yaml) is the prompting experiment used in the BMP loop.
For details, see Tabs. 6 - 8 of the supplementary. 


## Where are appropriate configs?

- **/configs** (this folder)
  - Hyperparameter configurations for the BMP loop experiments. Use these files to reproduce training and evaluation settings.

- **/mmpose/configs**
  - Configuration files for MMPose, following the same format and structure as MMPose v1.3.1. Supports models, datasets, and training pipelines.

- **/sam2/configs**
  - Configuration files for SAM2, matching the format and directory layout of the original SAM v2.1 repository. Use these for prompt-driven segmentation and related tasks.


