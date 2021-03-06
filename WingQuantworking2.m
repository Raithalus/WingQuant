%% Clear
clear all; close all; clc;
%% Load Images
WingFiles = dir('C:\Users\che7oz\Desktop\Wings_for_david_20171108current\K\*.tif');
for i=1:length(WingFiles);
    WingI = imread(WingFiles(i).name);
    WingI = WingI(:,:,1);
    WingIraw = WingI;
%     figure, imshow(WingI); title('Base Image'); %Base Image
%% Wing Identification
    WingI = imgaussfilt(WingI,2.5); %Gauss Stdev is manually adjusted
%     figure, imshow(WingI); title('Gauss');
    WingI = imcomplement(WingI); % Inverts bw img
    figure, imshow(WingI); title('Complement')
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
figure, imshow(bw2); %toggle2
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
% imshow(bw7)
% hold on
plot(centroids2(:,1), centroids2(:,2), 'b*')
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
%% Coordinates Insert
coord = [149	296	45	1099
191	283	45	1099
191	283	246	1348
221	134	380	1293
286	220	573	809
323	192	430	207
224	459	248	457
324	698	428	675
];



%% Path Length
for c = 1:8
figure, 
% imshow(bw7);hold on
% imshow(WingI1); hold on
r1 = coord(c,1);
c1 = coord(c,2);
r2 = coord(c,3);
c2 = coord(c,4);
hold on
plot(c1, r1, 'g*', 'MarkerSize', 15)
plot(c2, r2, 'g*', 'MarkerSize', 15)
hold off
D1 = bwdistgeodesic(bw7, c1, r1, 'quasi-euclidean');
D2 = bwdistgeodesic(bw7, c2, r2, 'quasi-euclidean');

D = D1 + D2;
D = round(D * 8) / 8;

D(isnan(D)) = inf;
skeleton_path = imregionalmin(D);
P = imoverlay(WingI1, imdilate(skeleton_path, ones(3,3)), [1 0 0]);
imshow(P, 'InitialMagnification', 200)
hold on
plot(c1, r1, 'g*', 'MarkerSize', 10)
plot(c2, r2, 'g*', 'MarkerSize', 10)
hold off

path_length = D(skeleton_path);
path_lengths(:,c) = path_length(1)
end
%%
% figure,
% P = imoverlay(bw2, imdilate(skeleton_path, ones(3,3)), [1 0 0]);
% imshow(P, 'InitialMagnification', 200)
% hold on
% plot(c1, r1, 'g*', 'MarkerSize', 10)
% plot(c2, r2, 'g*', 'MarkerSize', 10)
% hold off
%%
% WingIraw2 = double(WingIraw)
% bw2 = uint8(bw2)
% figure, imshow(bw2.*WingI1)
%%
% [rows cols] = find(skeleton_path<Inf) 

%% Rotate and Crop
% theta = -stats(WingID).Orientation;
% WingI = imrotate(WingI, theta);
% % figure, imshow(WingI)
% BW = edge(bw3,'canny');
% m0=find(sum(BW,1)>0);
% n0=find(sum(BW,2)>0);
% WingI1=imcrop(WingI,[min(m0),min(n0),max(m0)-min(m0),max(n0)-min(n0)]);
% figure, imshow(WingI1)
%% Edge Detection
% WingIedge = edge(bw2,'canny')
% WingIedge2 = uint8(WingIedge);
%%
% figure, 
% imshow(WingI); hold on;
% imshow(WingIedge2)
%% Good
% figure, 
% imshow(bw3.*WingIraw)
%%
%  BW2 = edge(DapiI1,'canny');
%     I2 = uint8(BW2);
% %     figure,imshow(255-I2*255);impixelregion;
%     m=find(sum(I2,1)>0);
%     w=find(sum(I2,2)>0);    
%     EW(i) = max(w)-min(w)+1;
%     Bins=50;
%     EL(i)=max(m)-min(m)+1;
%     L=floor(EL(i)/Bins);
%     y=1:Bins;z=1:Bins;centroidx1=1:Bins;centroidy1=1:Bins;
%     n=1:length(m);
%     for j=1:length(m),
%         n(j)=min(find(I2(:,m(j))>0));
%     end
%     n=smooth(n);n=smooth(n);n=smooth(n);
%     n=n';
%     EL(i)=max(m)-min(m)+1;
%     L=floor(EL(i)/Bins);
%     y=1:Bins;z=1:Bins;centroidx1=1:Bins;centroidy1=1:Bins;
%     gama=atan(gradient(n));
%     radius=20;
%     centroidx=round(m-sin(gama)*radius);
%     centroidy=round(n+cos(gama)*radius);
%     for j=1:Bins,
%         centroidx1(j)=centroidx(ceil(0.5*EL(i)-(Bins+1)/2*L+j*L));
%         centroidy1(j)=centroidy(ceil(0.5*EL(i)-(Bins+1)/2*L+j*L));
%         y(j)=mean2([I1(centroidy1(j),centroidx1(j)-4:centroidx1(j)+4) I1(centroidy1(j)-1,centroidx1(j)-4:centroidx1(j)+3)...
%             I1(centroidy1(j)+1,centroidx1(j)-3:centroidx1(j)+4) I1(centroidy1(j)-2,centroidx1(j)-3:centroidx1(j)+3) I1(centroidy1(j)+2,centroidx1(j)-3:centroidx1(j)+3)...
%             I1(centroidy1(j)-3,centroidx1(j)-2:centroidx1(j)+3) I1(centroidy1(j)+3,centroidx1(j)-3:centroidx1(j)+2) I1(centroidy1(j)-4,centroidx1(j)-2:centroidx1(j)+2)...
%             I1(centroidy1(j)+4,centroidx1(j)-2:centroidx1(j)+2)]);
%         z(j)=mean2([BcdI1(centroidy1(j),centroidx1(j)-4:centroidx1(j)+4) BcdI1(centroidy1(j)-1,centroidx1(j)-4:centroidx1(j)+3)...
%             BcdI1(centroidy1(j)+1,centroidx1(j)-3:centroidx1(j)+4) BcdI1(centroidy1(j)-2,centroidx1(j)-3:centroidx1(j)+3) BcdI1(centroidy1(j)+2,centroidx1(j)-3:centroidx1(j)+3)...
%             BcdI1(centroidy1(j)-3,centroidx1(j)-2:centroidx1(j)+3) BcdI1(centroidy1(j)+3,centroidx1(j)-3:centroidx1(j)+2) BcdI1(centroidy1(j)-4,centroidx1(j)-2:centroidx1(j)+2)...
%             BcdI1(centroidy1(j)+4,centroidx1(j)-2:centroidx1(j)+2)]);
%     end
% 

end %% Final