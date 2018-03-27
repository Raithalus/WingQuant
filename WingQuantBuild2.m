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
				figure, imshow(bw5); title('Smoothened; Extraneous Objects Removed')
				bw5 = uint8(bw5);
				bwt = double(~bw5);
				bwt = bwdist(bwt);
%% Skeleton
bw6 = bwmorph(bw5,'skel', Inf);
bw7 = bwmorph(bw6,'spur', 10);%Spur setting is manually adjusted
se = strel('disk',1);
bw7a = imdilate(bw7,se);
bw7 = bwmorph(bw7, 'thicken'); 
figure, imshow(bw7); title('Skeleton')
bw8 = bwmorph(bw7, 'branchpoints');
asdf = regionprops(bw8,'all');
centroids = cat(1, asdf.Centroid);
figure, imshow(bw7); hold on;
plot(centroids(:,1), centroids(:,2), '.g', 'MarkerSize', 20)
hold off
bw8 = bwmorph(bw8, 'thicken', 3); %Discriminates against certain obj
se = strel('disk',5);
bw8 = imdilate(bw8,se);
bw9 = bwmorph(bw7, 'endpoints');
asdf2 = regionprops(bw9,'all');
centroids2 = cat(1, asdf2.Centroid);
plot(centroids2(:,1), centroids2(:,2), '.b', 'MarkerSize', 20)
hold off
bw9 = imdilate(bw9,se);
%% Skeletal Overlay
% figure, 
% imshow(WingI1); title('Skeletal Overlay on Base Image');
% I = bw7a;
% J = bw8;
% K = bw9;
% red = cat(3, ones(size(WingI1)), zeros(size(WingI1)), zeros(size(WingI1))); 
green = cat(3, zeros(size(WingI1)), ones(size(WingI1)), zeros(size(WingI1)));
% blue = cat(3, zeros(size(WingI1)), zeros(size(WingI1)), ones(size(WingI1)));
% hold on ;
% h = imshow(red); 
% set(h, 'AlphaData', I);
% j = imshow(green); 
% set(j, 'AlphaData', J);
% k = imshow(blue);
% set(k, 'AlphaData', K);
% hold off;
%% Coordinate Array
coord = CoordArray.(CoordFields{i}); %Gets ith coordinates from CoordArray
% WingIDS
% s1 = WingIDS.Files{i,1}
% s2 = 
% strcmp()
%  if isequal(WingIDS.Files{i,1},'x')
% 	end
	

%% Path Length
for c = 1:8
% figure, 
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
% imshow(P, 'InitialMagnification', 200)
% hold on
% plot(c1, r1, '.g', 'MarkerSize', 20)
% plot(c2, r2, '.g', 'MarkerSize', 20)
% hold off

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
% figure, imshow(bwt,[]); hold on; 
% j = imshow(green); 
% plot(c1, r1, '.r', 'MarkerSize', 20)
% plot(c2, r2, '.b', 'MarkerSize', 20)
% set(j, 'AlphaData', diameterImage);
% hold off;
perimDI = bwperim(diameterImage);
[R, C] = find(perimDI, 1, 'first');
traceDI{:,:,c,i}= bwtraceboundary(perimDI,[R C], 'E', 8);

% figure, imshow(bwt,[]); hold on;
k = traceDI{:,:,c,i}(1:round(length(traceDI{:,:,c,i})/2),2);
j = traceDI{:,:,c,i}(1:round(length(traceDI{:,:,c,i})/2),1);
ChosenPixels{:, c, i} = improfile(bwt, k, j);




%% WIP Profile Alignment
% for i = 1:10
% L1max(i,1) = length(ChosenPixels{:,1,i})
% L1max = max(L1max)
% % midrow(i,:) = ChosenPixels{:,1,i}(ceil(end/2), :)
% L1mid = floor(L1max/2)
% 
% 
% end


%% Quick Fig

figure,
for i = 1:52
plot(ChosenPixels{:,1,i}); hold on; title('AllL1');
end
xlim([1, 1100])
export_fig('C:\Users\che7oz\Desktop\TempFigs\AllL1','-png');

figure,
for i = 1:52
plot(ChosenPixels{:,2,i}); hold on; title('AllL2');
end
xlim([1, 1100])
export_fig('C:\Users\che7oz\Desktop\TempFigs\AllL2','-png');

figure,
for i = 1:52
plot(ChosenPixels{:,3,i}); hold on; title('AllL3');
end
xlim([1, 1500])
export_fig('C:\Users\che7oz\Desktop\TempFigs\AllL3','-png');
figure,
for i = 1:52
plot(ChosenPixels{:,4,i}); hold on; title('AllL4');
end
xlim([1, 1500])
export_fig('C:\Users\che7oz\Desktop\TempFigs\AllL4','-png');
figure,
for i = 1:52
plot(ChosenPixels{:,5,i}); hold on; title('AllL5');
end
xlim([1, 800])
export_fig('C:\Users\che7oz\Desktop\TempFigs\AllL5','-png');
%% Quick Fig A
figure,
for i = 1:5
plot(ChosenPixels{:,1,i}); hold on; title('AL1');
end
xlim([1, 1100])
export_fig('C:\Users\che7oz\Desktop\TempFigs\AL1','-png');

figure,
for i = 1:5
plot(ChosenPixels{:,2,i}); hold on; title('AL2');
end
xlim([1, 1100])
export_fig('C:\Users\che7oz\Desktop\TempFigs\AL2','-png');

figure,
for i = 1:5
plot(ChosenPixels{:,3,i}); hold on; title('AL3');
end
xlim([1, 1500])
export_fig('C:\Users\che7oz\Desktop\TempFigs\AL3','-png');

figure,
for i = 1:5
plot(ChosenPixels{:,4,i}); hold on; title('AL4');
end
xlim([1, 1500])
export_fig('C:\Users\che7oz\Desktop\TempFigs\AL4','-png');

figure,
for i = 1:5
plot(ChosenPixels{:,5,i}); hold on; title('AL5');
end
xlim([1, 800])
export_fig('C:\Users\che7oz\Desktop\TempFigs\AL5','-png');
%% Quick Fig B
figure,
for i = 6:12
plot(ChosenPixels{:,1,i}); hold on; title('BL1');
end
xlim([1, 1100])
export_fig('C:\Users\che7oz\Desktop\TempFigs\BL1','-png');

figure,
for i = 6:12
plot(ChosenPixels{:,2,i}); hold on; title('BL2');
end
xlim([1, 1100])
export_fig('C:\Users\che7oz\Desktop\TempFigs\BL2','-png');

figure,
for i = 6:12
plot(ChosenPixels{:,3,i}); hold on; title('BL3');
end
xlim([1, 1500])
export_fig('C:\Users\che7oz\Desktop\TempFigs\BL3','-png');

figure,
for i = 6:12
plot(ChosenPixels{:,4,i}); hold on; title('BL4');
end
xlim([1, 1500])
export_fig('C:\Users\che7oz\Desktop\TempFigs\BL4','-png');

figure,
for i = 6:12
plot(ChosenPixels{:,5,i}); hold on; title('BL5');
end
xlim([1, 800])
export_fig('C:\Users\che7oz\Desktop\TempFigs\BL5','-png');
%% Quick Fig C
figure,
for i = 13:24
plot(ChosenPixels{:,1,i}); hold on; title('CL1');
end
xlim([1, 1100])
export_fig('C:\Users\che7oz\Desktop\TempFigs\CL1','-png');

figure,
for i = 13:24
plot(ChosenPixels{:,2,i}); hold on; title('CL2');
end
xlim([1, 1100])
export_fig('C:\Users\che7oz\Desktop\TempFigs\CL2','-png');

figure,
for i = 13:24
plot(ChosenPixels{:,3,i}); hold on; title('CL3');
end
xlim([1, 1500])
export_fig('C:\Users\che7oz\Desktop\TempFigs\CL3','-png');

figure,
for i = 13:24
plot(ChosenPixels{:,4,i}); hold on; title('CL4');
end
xlim([1, 1500])
export_fig('C:\Users\che7oz\Desktop\TempFigs\CL4','-png');

figure,
for i = 13:24
plot(ChosenPixels{:,5,i}); hold on; title('CL5');
end
xlim([1, 800])
export_fig('C:\Users\che7oz\Desktop\TempFigs\CL5','-png');
%% Quick Fig D
figure,
for i = 25:31
plot(ChosenPixels{:,1,i}); hold on; title('DL1');
end
xlim([1, 1100])
export_fig('C:\Users\che7oz\Desktop\TempFigs\DL1','-png');

figure,
for i = 25:31
plot(ChosenPixels{:,2,i}); hold on; title('DL2');
end
xlim([1, 1100])
export_fig('C:\Users\che7oz\Desktop\TempFigs\DL2','-png');

figure,
for i = 25:31
plot(ChosenPixels{:,3,i}); hold on; title('DL3');
end
xlim([1, 1500])
export_fig('C:\Users\che7oz\Desktop\TempFigs\DL3','-png');

figure,
for i = 25:31
plot(ChosenPixels{:,4,i}); hold on; title('DL4');
end
xlim([1, 1500])
export_fig('C:\Users\che7oz\Desktop\TempFigs\DL4','-png');

figure,
for i = 25:31
plot(ChosenPixels{:,5,i}); hold on; title('DL5');
end
xlim([1, 800])
export_fig('C:\Users\che7oz\Desktop\TempFigs\DL5','-png');
%% Quick Fig E
figure,
for i = 32:35
plot(ChosenPixels{:,1,i}); hold on; title('EL1');
end
xlim([1, 1100])
export_fig('C:\Users\che7oz\Desktop\TempFigs\EL1','-png');

figure,
for i = 32:35
plot(ChosenPixels{:,2,i}); hold on; title('EL2');
end
xlim([1, 1100])
export_fig('C:\Users\che7oz\Desktop\TempFigs\EL2','-png');

figure,
for i = 32:35
plot(ChosenPixels{:,3,i}); hold on; title('EL3');
end
xlim([1, 1500])
export_fig('C:\Users\che7oz\Desktop\TempFigs\EL3','-png');

figure,
for i = 32:35
plot(ChosenPixels{:,4,i}); hold on; title('EL4');
end
xlim([1, 1500])
export_fig('C:\Users\che7oz\Desktop\TempFigs\EL4','-png');

figure,
for i = 32:35
plot(ChosenPixels{:,5,i}); hold on; title('EL5');
end
xlim([1, 800])
export_fig('C:\Users\che7oz\Desktop\TempFigs\EL5','-png');
%% Quick Fig F
figure,
for i = 36:45
plot(ChosenPixels{:,1,i}); hold on; title('FL1');
end
xlim([1, 1100])
export_fig('C:\Users\che7oz\Desktop\TempFigs\FL1','-png');

figure,
for i = 36:45
plot(ChosenPixels{:,2,i}); hold on; title('FL2');
end
xlim([1, 1100])
export_fig('C:\Users\che7oz\Desktop\TempFigs\FL2','-png');

figure,
for i = 36:45
plot(ChosenPixels{:,3,i}); hold on; title('FL3');
end
xlim([1, 1500])
export_fig('C:\Users\che7oz\Desktop\TempFigs\FL3','-png');

figure,
for i = 36:45
plot(ChosenPixels{:,4,i}); hold on; title('FL4');
end
xlim([1, 1500])
export_fig('C:\Users\che7oz\Desktop\TempFigs\FL4','-png');

figure,
for i = 36:45
plot(ChosenPixels{:,5,i}); hold on; title('FL5');
end
xlim([1, 800])
export_fig('C:\Users\che7oz\Desktop\TempFigs\FL5','-png');
%%
% %% Quick Fig G (missing data)
% figure,
% for i = 46:52
% plot(ChosenPixels{:,1,i}); hold on; title('GL1');
% end
% xlim([1, 1100])
% 
% figure,
% for i = 46:52
% plot(ChosenPixels{:,2,i}); hold on; title('GL2');
% end
% xlim([1, 1100])
% 
% figure,
% for i = 46:52
% plot(ChosenPixels{:,3,i}); hold on; title('GL3');
% end
% xlim([1, 1500])
% 
% figure,
% for i = 46:52
% plot(ChosenPixels{:,4,i}); hold on; title('GL4');
% end
% xlim([1, 1500])
% 
% figure,
% for i = 46:52
% plot(ChosenPixels{:,5,i}); hold on; title('GL5');
% end
% xlim([1, 800])
%% 
% I = perimDI
% figure,
% imshow(I)
% [d, b] = find(I, 1, 'first');
% mycontour = bwtraceboundary(I, [d b] ,'E',8,Inf,'counterclockwise') ;I = perimDI
% figure,
% imshow(I)
% [r, d] = find(I, 1, 'first');
% mycontour = bwtraceboundary(I, [r d] ,'E',8,Inf,'counterclockwise') ;
% figure,
% imshow(bwt,[])
% % export_fig bwdistex.png
% hold on;
% plot(mycontour(:,2),mycontour(:,1),'g','LineWidth',1);
% k = mycontour(1:round(length(mycontour)/2),2);
% j = mycontour(1:round(length(mycontour)/2),1);
% ChosenPixels(:,:,c,i) = improfile(bwt, k, j)
% figure, plot(ChosenPixels)

end
%% Vein Diameter: Veins ACV/PCV
% for c = 7:8
% r1 = coord(c,1);
% c1 = coord(c,2);
% r2 = coord(c,3);
% c2 = coord(c,4);
% D1 = bwdistgeodesic(bw7, c1, r1, 'quasi-euclidean');
% D2 = bwdistgeodesic(bw7, c2, r2, 'quasi-euclidean');
% D = D1 + D2;
% D = round(D * 8) / 8;
% D(isnan(D)) = inf;
% skeleton_path = imregionalmin(D);
% bw5 = bwmorph(bw4, 'spur', Inf);
% edtImage = 2 * bwdist(~bw5);
% diameterImage = edtImage .* double(skeleton_path);
% figure, imshow(bwt,[]); hold on; 
% j = imshow(green); 
% plot(c1, r1, '.r')
% plot(c2, r2, '.b')
% set(j, 'AlphaData', diameterImage);
% perimDI = bwperim(diameterImage);
% [r, c] = find(perimDI, 1, 'first');
% traceDI = bwtraceboundary(perimDI,[c r], 'S', 8,Inf,'counterclockwise');
% plot(traceDI(:,2),traceDI(:,1),'g','LineWidth',1);
end
%%
% vbw = imread('E:\Wings_for_david_20171108current\Wing Vein bw\E4.png');
% vbw = im2bw(vbw);
% bwt = double(~bw5);
% bwt = bwdist(bwt);
% figure, imshow(bwt,[]);
% export_fig('BWDist.png')

% end
%%
% figure, imshow(diameterImage);hold on; scatter(c1, r1,'*r')


%%  Example Works
% I = perimDI
% figure,
% imshow(I)
% [r, c] = find(I, 1, 'first');
% mycontour = bwtraceboundary(I, [r c] ,'E',8,Inf,'counterclockwise') ;
% figure,
% imshow(bwt,[])
% % export_fig bwdistex.png
% hold on;
% plot(mycontour(:,2),mycontour(:,1),'g','LineWidth',1);
% i = mycontour(1:round(length(mycontour)/2),2);
% j = mycontour(1:round(length(mycontour)/2),1);
% ChosenPixels(:,:) = improfile(bwt, i, j)
% figure, plot(ChosenPixels)