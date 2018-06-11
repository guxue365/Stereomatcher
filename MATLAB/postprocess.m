clc, clear, close all;

I = im2double(imread('img.png'));

I2 = I>0.3 & I<0.4;

imshow(I2);