im = imread('42_meitu_2.jpg');
imshow(im)

lab = RGB2Lab(im);
saliency = zeros(size(im,1),size(im,2));

for i = 1:size(im,1)
   for j = 1:size(im,2)
      diffSum = 0;
      for ii = 1:size(im,1)
           for jj = 1:size(im,2)
              theDiff = lab(i,j,:)-lab(ii,jj,:);
              diff = [theDiff(1),theDiff(2),theDiff(3)];
              diffNorm = norm(diff);
              diffSum = diffSum + diffNorm;
              
           end
      end
      saliency(i,j) = diffSum;
      
   end
end
normalize it
theMax = max(saliency(:));
saliency = saliency / theMax;

figure
imshow(saliency)

saliencySorted = sort(saliency(:),'descend');
percentageThreshold = 0.12;
threshold = saliencySorted(floor(percentageThreshold*length(saliencySorted)));

BW = im2bw(saliency, threshold);
figure
imshow(BW)
se = strel('disk',2);        
BW = imdilate(BW,se);
BW = imerode(BW,se);
figure
imshow(BW)

%find the largest one
CC = bwconncomp(BW);
numPixels = cellfun(@numel,CC.PixelIdxList);
[biggest,idx] = max(numPixels);
BW(:) = 0;
BW(CC.PixelIdxList{idx}) = 1;
figure
imshow(BW)

STATS = regionprops(BW, 'BoundingBox');
b = STATS.BoundingBox;
boundingbox = [ceil(b(2)),ceil(b(1)),floor(b(4)),floor(b(3))];
boundingbox(3) = boundingbox(1) + boundingbox(3) - 1;
boundingbox(4) = boundingbox(2) + boundingbox(4) - 1;
finalIm = drawRectangleOnImage(im,boundingbox);
figure
imshow(finalIm)