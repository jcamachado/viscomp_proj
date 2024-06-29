close all
clear
clc

% Input image
f = im2double(imread('cameraman.tif'));

fig = figure('Color',[1,1,1]);
imagesc(f)
axis equal
axis off
colormap gray
colorbar
saveas(fig,'input.png')

% Sobel X
sx = [-1 0 1
      -2 0 2
      -1 0 1];
  
jx = imfilter(f,sx,'conv');

fig = figure('Color',[1,1,1]);
imagesc(jx)
axis equal
axis off
colormap gray
colorbar
saveas(fig,'sobel_x.png')

% Sobel Y
sy = [-1 -2 -1
       0  0  0
       1  2  1];
   
jy = imfilter(f,sy,'conv');

fig = figure('Color',[1,1,1]);
imagesc(jy)
axis equal
axis off
colormap gray
colorbar
saveas(fig,'sobel_y.png')

sy = [-1 -2 -1
       0  0  0
       1  2  1];
   
jy = imfilter(f,sy,'conv');

% Squared magnitude
g = jx.^2 + jy.^2;

fig = figure('Color',[1,1,1]);
imagesc(g)
axis equal
axis off
colormap gray
colorbar
saveas(fig,'magnitude_sqr.png')

% Laplacian
h = [ 0 -1  0
     -1  4 -1
      0 -1  0];
   
l = imfilter(f,h,'conv');

fig = figure('Color',[1,1,1]);
imagesc(l)
axis equal
axis off
colormap gray
colorbar
saveas(fig,'laplacian.png')

% Laplacian of Gaussian
h = fspecial('log',5,0.5);
   
l = imfilter(f,h,'conv');

fig = figure('Color',[1,1,1]);
imagesc(l)
axis equal
axis off
colormap gray
colorbar
saveas(fig,'laplacian_of_gaussian.png')