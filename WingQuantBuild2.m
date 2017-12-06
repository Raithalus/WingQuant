%% Clear
clear all; close all; clc;
%% Loading
run CoordinateArray.m %Manually curated coordinates
WingIDS = imageDatastore('C:\Users\che7oz\Desktop\Wing Quant\Wing Images\','FileExtensions','.tif', 'LabelSource', 'foldernames','IncludeSubfolders', 1)
%% Load Images
for i=1:length(WingIDS.Files);
				WingI = readDatastoreImage(WingIDS.Files{i,1});
    WingI = WingI(:,:,1);
    WingIraw = WingI;
%% Wing Identification
    WingI = imgaussfilt(WingI,2.5); %Gauss Stdev is manually adjusted
%     figure, imshow(WingI); title('Gauss');
    WingI = imcomplement(WingI); % Inverts bw img
%     figure, imshow(WingI); title('Complement')
    WingIthresh = graythresh(WingI);
    WingIthreshback(i,:) = WingIthresh(:);
    bw = imbinarize(WingI, 'Adaptive','ForegroundPolarity','bright','Sensitivity',0.4); %Sensitivity setting is manually adjusted
%     figure, imshow (bw); title('Binary Image'); %toggle1
    bw = bwmorph(bw, 'fill');
    [N,M] = bwlabel(bw,4);
				stats = regionprops(N,'all');
				WingArea = [stats.Area];
				[WingSize, WingID] = max(WingArea);
				N(find(N~=WingID))=0;
				NN = (N~=0);
				bw2 = bw.*NN;
				BW = edge(bw2,'canny');
				m0=find(sum(BW,1)>0);
				n0=find(sum(BW,2)>0);
				WingI1=imcrop(WingIraw,[min(m0),min(n0),max(m0)-min(m0),max(n0)-min(n0)]);
				bw2=imcrop(bw2,[min(m0),min(n0),max(m0)-min(m0),max(n0)-min(n0)]);
% 				figure, imshow(bw2); %toggle2
				bw3 = bwmorph(bw2, 'open', Inf);
				bw4 = bwmorph(bw3, 'close', Inf);
				bw5 = bwmorph(bw4, 'spur', Inf);
				figure, imshow(bw5); title('Smoothened; Extraneous Objects Removed')
				bw5 = uint8(bw5);
%% Skeleton
bw6 = bwmorph(bw5,'skel', Inf);
bw7 = bwmorph(bw6,'spur', 10);%Spur setting is manually adjusted
se = strel('disk',1);
bw7a = imdilate(bw7,se);
% bw7 = bwmorph(bw7, 'thicken'); 
figure, imshow(bw7); title('Skeleton')
bw8 = bwmorph(bw7, 'branchpoints');
asdf = regionprops(bw8,'all');
centroids = cat(1, asdf.Centroid);
imshow(bw7); hold on;
plot(centroids(:,1), centroids(:,2), 'g*')
% hold off
% bw8 = bwmorph(bw8, 'thicken', 3); %Discriminates against certain obj
se = strel('disk',5);
bw8 = imdilate(bw8,se);
bw9 = bwmorph(bw7, 'endpoints');
asdf2 = regionprops(bw9,'all');
centroids2 = cat(1, asdf2.Centroid);
plot(centroids2(:,1), centroids2(:,2), 'b*')
hold off
bw9 = imdilate(bw9,se);
end