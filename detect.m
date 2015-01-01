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
%normalize it
theMax = max(saliency(:));
saliency = saliency / theMax;

figure
imshow(saliency)