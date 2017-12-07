%% Clear
clear all; close all; clc;
format longg;
format compact;
%% Loading
run CoordinateArray.m %Manually curated coordinates
CoordFields = fieldnames(CoordArray);
WingIDS = imageDatastore('C:\Users\che7oz\Desktop\WingDB','FileExtensions','.tif', 'LabelSource', 'foldernames','IncludeSubfolders', 1)
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
						
				if ismember(i,36:45)==1
								bw = imbinarize(WingI, 'Adaptive','ForegroundPolarity','bright','Sensitivity',0.4); %Sensitivity setting is manually adjusted; F Thresh is different
				else
								bw = imbinarize(WingI, 'Adaptive','ForegroundPolarity','bright','Sensitivity',0.315); %Sensitivity setting is manually adjusted
				end
				
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
% 				figure, imshow(bw5); title('Smoothened; Extraneous Objects Removed')
				bw5 = uint8(bw5);
				bwt = double(~bw5);
				bwt = bwdist(bwt);
%% Skeleton
bw6 = bwmorph(bw5,'skel', Inf);
bw7 = bwmorph(bw6,'spur', 10);%Spur setting is manually adjusted
se = strel('disk',1);
bw7a = imdilate(bw7,se);
% bw7 = bwmorph(bw7, 'thicken'); 
% figure, imshow(bw7); title('Skeleton')
bw8 = bwmorph(bw7, 'branchpoints');
asdf = regionprops(bw8,'all');
centroids = cat(1, asdf.Centroid);
figure, imshow(bw7); hold on;
plot(centroids(:,1), centroids(:,2), '.g', 'MarkerSize', 20)
% hold off
% bw8 = bwmorph(bw8, 'thicken', 3); %Discriminates against certain obj
se = strel('disk',5);
bw8 = imdilate(bw8,se);
bw9 = bwmorph(bw7, 'endpoints');
asdf2 = regionprops(bw9,'all');
centroids2 = cat(1, asdf2.Centroid);
plot(centroids2(:,1), centroids2(:,2), '.b', 'MarkerSize', 20)
hold off
bw9 = imdilate(bw9,se);
%% Skeletal Overlay
figure, 
imshow(WingI1); title('Skeletal Overlay on Base Image');
I = bw7a;
J = bw8;
K = bw9;
red = cat(3, ones(size(WingI1)), zeros(size(WingI1)), zeros(size(WingI1))); 
green = cat(3, zeros(size(WingI1)), ones(size(WingI1)), zeros(size(WingI1)));
blue = cat(3, zeros(size(WingI1)), zeros(size(WingI1)), ones(size(WingI1)));
hold on ;
h = imshow(red); 
set(h, 'AlphaData', I);
j = imshow(green); 
set(j, 'AlphaData', J);
k = imshow(blue);
set(k, 'AlphaData', K);
hold off;
%% Coordinate Array
coord = CoordArray.(CoordFields{i}) %Gets ith coordinates from CoordArray
% WingIDS
% s1 = WingIDS.Files{i,1}
% s2 = 
% strcmp()
%  if isequal(WingIDS.Files{i,1},'x')
% 	end
	

%% Path Length
for c = 1:8
figure, 
% imshow(bw7);hold on
% imshow(WingI1); hold on
r1 = coord(c,1);
c1 = coord(c,2);
r2 = coord(c,3);
c2 = coord(c,4);
% hold on
% plot(c1, r1, 'g*', 'MarkerSize', 15)
% plot(c2, r2, 'g*', 'MarkerSize', 15)
% hold off
D1 = bwdistgeodesic(bw7, c1, r1, 'quasi-euclidean');
D2 = bwdistgeodesic(bw7, c2, r2, 'quasi-euclidean');

D = D1 + D2;
D = round(D * 8) / 8;

D(isnan(D)) = inf;
skeleton_path = imregionalmin(D);
P = imoverlay(WingI1, imdilate(skeleton_path, ones(3,3)), [1 0 0]);
imshow(P, 'InitialMagnification', 200)
hold on
plot(c1, r1, '.g', 'MarkerSize', 20)
plot(c2, r2, '.g', 'MarkerSize', 20)
hold off

path_length = D(skeleton_path);
path_lengths(i,c) = path_length(1);
end
%% Vein Diameter: Veins 1-6
% vbw = imread('C:\Users\che7oz\Desktop\Wings_for_david_20171108current\E\E4.png');
% vbw = imcrop(vbw,[min(m0),min(n0),max(m0)-min(m0),max(n0)-min(n0)]);
VeinWidth = zeros(8,2000);
for c = 1:6
r1 = coord(c,1);
c1 = coord(c,2);
r2 = coord(c,3);
c2 = coord(c,4);
D1 = bwdistgeodesic(bw7, c1, r1, 'quasi-euclidean');
D2 = bwdistgeodesic(bw7, c2, r2, 'quasi-euclidean');
D = D1 + D2;
D = round(D * 8) / 8;
D(isnan(D)) = inf;
skeleton_path = imregionalmin(D);
bw5 = bwmorph(bw4, 'spur', Inf);
% bw5 = vbw;
edtImage = 2 * bwdist(~bw5); %figure, imshow(edtImage);
diameterImage = edtImage .* double(skeleton_path);
figure, imshow(bwt,[]); hold on; 
j = imshow(green); 
plot(c1, r1, '.r', 'MarkerSize', 20)
plot(c2, r2, '.b', 'MarkerSize', 20)
set(j, 'AlphaData', diameterImage);
perimDI = bwperim(diameterImage);
traceDI = bwtraceboundary(diameterImage,[c1 r1], 'E', 8);
hold off;
end
%% Vein Diameter: Veins ACV/PCV
for c = 7:8
r1 = coord(c,1);
c1 = coord(c,2);
r2 = coord(c,3);
c2 = coord(c,4);
D1 = bwdistgeodesic(bw7, c1, r1, 'quasi-euclidean');
D2 = bwdistgeodesic(bw7, c2, r2, 'quasi-euclidean');
D = D1 + D2;
D = round(D * 8) / 8;
D(isnan(D)) = inf;
skeleton_path = imregionalmin(D);
bw5 = bwmorph(bw4, 'spur', Inf);
edtImage = 2 * bwdist(~bw5);
diameterImage = edtImage .* double(skeleton_path);
figure, imshow(bwt,[]); hold on; 
j = imshow(green); 
plot(c1, r1, '.r')
plot(c2, r2, '.b')
set(j, 'AlphaData', diameterImage);
perimDI = bwperim(diameterImage);
traceDI = bwtraceboundary(diameterImage,[c1 r1], 'E', 8);
end
% VeinWidth = VeinWidth';
%%
% vbw = imread('E:\Wings_for_david_20171108current\Wing Vein bw\E4.png');
% vbw = im2bw(vbw);
% bwt = double(~bw5);
% bwt = bwdist(bwt);
% figure, imshow(bwt,[]);
% export_fig('BWDist.png')

end

figure, imshow(diameterImage)