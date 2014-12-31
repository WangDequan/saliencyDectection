im = imread('42.jpg');
imshow(im)
diffIm = im;
lab = RGB2Lab(im);
imshow(lab)

% for i = 1:size(im,1)
%    for j = 1:size(im,2)
%       theDiff = 0;
%       for ii = 1:size(im,1)
%            for jj = 1:size(im,2)
%               if(i ~= ii || j ~= jj)
%                   
%               end
%            end
%       end
%       
%    end
% end