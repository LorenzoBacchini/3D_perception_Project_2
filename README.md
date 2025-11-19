# Introduction
This repo contains the code developed for the second project of my 3D perception university course

# Content
The code provided in the repo provide improved gound segmentation fusing data from LiDAR and camera without using machine learning for the fusion phase

# Dataset
As you can see in the repo i used the KITTI dataset, in particular the "raw KITTI" dataset, modifying a bit the dataset structure and maintaining only the useful folders and files.
In the solution provided the sensors used are the left camera and the LiDAR

# Implementation
In the solution developed there are three main files, ground_plane_segmentation.py (used to perform the image and point cloud segmentations), kitti_dataloader.py (to bootstrap the dataset files) and fusion.py (contains the 
main to launch the segmentation pipeline)
in fusion_utils.py you can find simply some functions used by fusion.py

## Segmentation
To perform the segmentation two different approach were applied, for the camera segmentation has been used a U-Net trained on the Cityscape dataset, while for the point cloud segmentation has been 
implemented the RANSAC algorithm.

# Fusion
the fusion is a simply weighted fusion that trusts the camera more as it move up in the image and viceversa for the LiDAR (because the LiDAR point become more sparse in the upper part of the image) 

# Testing
To test the entire pipeline just clone the repo and launch in the terminal the fusion.py file
> [!NOTE]
> if you want to test the pipeline on your dataset, simply update the parameter in the main of fusion.py file
