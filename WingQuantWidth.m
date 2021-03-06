%% Clear
clear all; close all; clc;
%% Load Images
WingFiles = dir('C:\Users\David Cheung\Desktop\Wings_for_david_20171108current\B\*.tif');
for i=1:length(WingFiles);
    WingI = imread(WingFiles(i).name);
    WingI = WingI(:,:,1);
    WingIraw = WingI;
%     figure, imshow(WingI); title('Base Image'); %Base Image
%% Wing Identification
    WingI = imgaussfilt(WingI,2.5); %Gauss Stdev is manually adjusted
%     figure, imshow(WingI); title('Gauss');
    WingI = imcomplement(WingI); % Inverts bw img
%     figure, imshow(WingI); title('Complement')
    bw = imbinarize(WingI, 'Adaptive','ForegroundPolarity','bright','Sensitivity',0.315); %Sensitivity setting is manually adjusted
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
imshow(WingI1); %title('Skeletal Overlay on Base Image');
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
export_fig('Overlay.png')
%% Coordinates Insert
coord = [131	119	78	1091
156	236	78	1091
156	236	318	1345
179	81	456	1273
253	160	610	742
253	160	360	143
210	436	240	425
341	664	448	617


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
plot(c1, r1, 'g.', 'MarkerSize', 15)
plot(c2, r2, 'g.', 'MarkerSize', 15)
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
plot(c1, r1, 'g.', 'MarkerSize', 10)
plot(c2, r2, 'g.', 'MarkerSize', 10)
hold off

path_length = D(skeleton_path);
path_lengths(:,c) = path_length(1);
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
figure, imshow(diameterImage); hold on; plot(c1, r1, '*r')
perimDI = bwperim(diameterImage);
traceDI = bwtraceboundary(diameterImage,[c1-1 r1-1], 'E', 8);


% ysum = sum(diameterImage);
% ysum = double(ysum(ysum>0));
% figure, plot(ysum);
% VeinWidth(c,1:length(ysum)) = ysum;
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
figure, imshow(diameterImage);
% ysum = sum(diameterImage'); %addition is directoinal FIX ME
% figure, plot(ysum)
% ysum = ysum(ysum>0);
% figure, plot(ysum);
% VeinWidth(c,1:length(ysum)) = ysum;
end
% VeinWidth = VeinWidth';
%%
vbw = imread('E:\Wings_for_david_20171108current\Wing Vein bw\E4.png');
vbw = im2bw(vbw);
bwt = double(~bw5);
bwt = bwdist(bwt);
figure, imshow(bwt,[]);
export_fig('BWDist.png')

end %% Final

%% Example
BW = perimDI
       imshow(BW,[]);
       s=size(BW);
       for row = 1:s(1)
         for col=1:s(2)
           if BW(row,col), 
             break;
           end
         end
 
         contour = bwtraceboundary(BW, [row, col], 'E', 8);
         if(~isempty(contour))
           hold on; plot(contour(:,2),contour(:,1),'g','LineWidth',2);
           hold on; plot(col, row,'gx','LineWidth',2);
         else
           hold on; plot(col, row,'rx','LineWidth',2);
         end
       end
       
%%  Example Works
I = perimDI
figure(1)
imshow(I)
[r, c] = find(I, 1, 'first');
mycontour = bwtraceboundary(I, [r c] ,'E',8,Inf,'counterclockwise') ;
figure,
imshow(bwt,[])
% export_fig bwdistex.png
hold on;
plot(mycontour(:,2),mycontour(:,1),'g','LineWidth',1);
%%
i = mycontour(1:round(length(mycontour)/2),2);
j = mycontour(1:round(length(mycontour)/2),1);
ChosenPixels(:,:) = improfile(bwt, i, j)
figure, plot(ChosenPixels)