clc, clear, close all;

addpath('E:\dataset_kitti\devkit\matlab');

IL = 256*rgb2gray(disp_read('E:/dataset_kitti/data_scene_flow/training/image_2/000000_10.png'));
IR = 256*rgb2gray(disp_read('E:/dataset_kitti/data_scene_flow/training/image_3/000000_10.png'));
GT = disp_read('E:/dataset_kitti/data_scene_flow/training/disp_noc_0/000000_10.png');

IL = IL(100:299, 200:399);
IR = IR(100:299, 200:399);
GT = GT(100:299, 200:399);