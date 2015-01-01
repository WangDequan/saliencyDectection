clc;clear;
addpath('RGB2Lab')

im = imread('42.jpg');
imshow(im)

quant_im = zeros(size(im));
for k = 1:3
    for i = 1:size(quant_im,1) 
        for j = 1:size(quant_im,2)
            quant_im(i,j,k) = ceil(im(i,j,k) / 21.25);
        end
    end
end
quant_im = uint8(quant_im);
quant_im(quant_im==0) = 1;

%show the image after quantization
quant_im_minus1 = quant_im - 1;
quant_im_show = quant_im_minus1 * 23;
quant_im_show(quant_im_show==253)= 255;
figure
imshow(quant_im_show)

lab = RGB2Lab(quant_im_show);
%begin to count
countSpace = zeros(12,12,12);
saliencySpace = zeros(12,12,12);
labSpace = cell(12,12,12);
changeColorSpace = cell(12,12,12);%this part mark if the colour space need to be change
for i = 1:12
    for j = 1:12
        for k = 1:12
            changeColorSpace{i,j,k} = {i,j,k};
        end
    end
end
for i = 1:size(im,1)
    for j = 1:size(im,2)
        r_level = quant_im(i,j,1);
        g_level = quant_im(i,j,2);
        b_level = quant_im(i,j,3);
        countSpace(r_level,g_level,b_level) = countSpace(r_level,g_level,b_level) + 1;
        labSpace{r_level,g_level,b_level} = {lab(i,j,1),lab(i,j,2),lab(i,j,3)};
     
    end
end

[sortedArray,rank] = sort(countSpace(:),'descend');
totalPixelNum = size(im,1) * size(im,2);
sumPixel = 0;
pixelThreshold = 0.99;
totalPixelThreshold = floor(totalPixelNum * pixelThreshold);
for i = 1:length(sortedArray)
    sumPixel = sumPixel + sortedArray(i);
    if(sumPixel > totalPixelThreshold)
        
        break
    end
end

A = zeros(2,2,3);
A(:,:,1) = [1,2;3,4];
A(:,:,2) = [5,6;7,8];
A(:,:,3) = [9,10;11,12];

IND = [3 4;5 6];
s = [2,2,2];
[I,J,K] = ind2sub(s,IND)

% saliency = zeros(size(im,1),size(im,2));
% 
% for i = 1:size(im,1)
%    for j = 1:size(im,2)
%       diffSum = 0;
%       for ii = 1:size(im,1)
%            for jj = 1:size(im,2)
%               theDiff = lab(i,j,:)-lab(ii,jj,:);
%               diff = [theDiff(1),theDiff(2),theDiff(3)];
%               diffNorm = norm(diff);
%               diffSum = diffSum + diffNorm;
%               
%            end
%       end
%       saliency(i,j) = diffSum;
%       
%    end
% end
% normalize it
% theMax = max(saliency(:));
% saliency = saliency / theMax;
% 
% figure
% imshow(saliency)
% 
% saliencySorted = sort(saliency(:),'descend');
% percentageThreshold = 0.12;
% threshold = saliencySorted(floor(percentageThreshold*length(saliencySorted)));
% 
% BW = im2bw(saliency, threshold);
% figure
% imshow(BW)
% se = strel('disk',2);        
% BW = imdilate(BW,se);
% BW = imerode(BW,se);
% figure
% imshow(BW)
% 
% %find the largest one
% CC = bwconncomp(BW);
% numPixels = cellfun(@numel,CC.PixelIdxList);
% [biggest,idx] = max(numPixels);
% BW(:) = 0;
% BW(CC.PixelIdxList{idx}) = 1;
% figure
% imshow(BW)
% 
% STATS = regionprops(BW, 'BoundingBox');
% b = STATS.BoundingBox;
% boundingbox = [ceil(b(2)),ceil(b(1)),floor(b(4)),floor(b(3))];
% boundingbox(3) = boundingbox(1) + boundingbox(3) - 1;
% boundingbox(4) = boundingbox(2) + boundingbox(4) - 1;
% finalIm = drawRectangleOnImage(im,boundingbox);
% figure
% imshow(finalIm)