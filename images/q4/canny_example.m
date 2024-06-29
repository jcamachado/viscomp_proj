close all
clear
clc

f = im2double(imread('cameraman.tif'));

fig = figure('Color',[1,1,1]);
imagesc(f)
axis equal
axis off
colormap gray
colorbar
saveas(fig,'input.png')

e = edge(f,'zerocross');

fig = figure('Color',[1,1,1]);
imagesc(e)
axis equal
axis off
colormap gray
colorbar
saveas(fig,'zerocross.png')

e = edge(f,'Canny');

fig = figure('Color',[1,1,1]);
imagesc(e)
axis equal
axis off
colormap gray
colorbar
saveas(fig,'canny.png')
