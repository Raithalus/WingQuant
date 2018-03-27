%% Clear
% clear all; close all; format longg; format compact; clc;
%% 
run CoordArrayCurated.m
%%
WingIDS = imageDatastore('C:\Users\che7oz\Desktop\Wing DB Curated','FileExtensions','.tif', 'LabelSource', 'foldernames','IncludeSubfolders', 1);
%% Label Count
WingLabel = countEachLabel(WingIDS);
A = 1:WingLabel.Count(1);
B = max(A)+1:WingLabel.Count(2)+max(A);
C = max(B)+1:WingLabel.Count(3)+max(B);
D = max(C)+1:WingLabel.Count(4)+max(C);
E = max(D)+1:WingLabel.Count(5)+max(D);
F = max(E)+1:WingLabel.Count(6)+max(E);
G = max(F)+1:WingLabel.Count(7)+max(F);
H = max(G)+1:WingLabel.Count(8)+max(G);
I = max(H)+1:WingLabel.Count(9)+max(H);
J = max(I)+1:WingLabel.Count(10)+max(I);
K = max(J)+1:WingLabel.Count(11)+max(J);
L = max(K)+1:WingLabel.Count(12)+max(K);
M = max(L)+1:WingLabel.Count(13)+max(L);
N = max(M)+1:WingLabel.Count(14)+max(M);
O = max(N)+1:WingLabel.Count(15)+max(N);
P = max(O)+1:WingLabel.Count(16)+max(O);
Q = max(P)+1:WingLabel.Count(17)+max(P);
R = max(Q)+1:WingLabel.Count(18)+max(Q);
S = max(R)+1:WingLabel.Count(19)+max(R);
T = max(S)+1:WingLabel.Count(20)+max(S);
U = max(T)+1:WingLabel.Count(21)+max(T);
V = max(U)+1:WingLabel.Count(22)+max(U);
W = max(V)+1:WingLabel.Count(23)+max(V);
X = max(W)+1:WingLabel.Count(24)+max(W);
Y = max(X)+1:WingLabel.Count(25)+max(X);
Z = max(Y)+1:WingLabel.Count(26)+max(Y);
AA = max(Z)+1:WingLabel.Count(27)+max(Z);
AB = max(AA)+1:WingLabel.Count(28)+max(AA);
AC = max(AB)+1:WingLabel.Count(29)+max(AB);
AD = max(AC)+1:WingLabel.Count(30)+max(AC);
AE = max(AD)+1:WingLabel.Count(31)+max(AD);
AF = max(AE)+1:WingLabel.Count(32)+max(AE);
AG = max(AF)+1:WingLabel.Count(33)+max(AF);
AH = max(AG)+1:WingLabel.Count(34)+max(AG);
AI = max(AH)+1:WingLabel.Count(35)+max(AH);



%%
AlphaNum = {char('A'); 'B'; 'C'; 'D'; 'E'; 'F'; 'G'; 'H'; 'I'; 'J'; 'K'; 'L'; 'M'; 'N'; 'O'; 'P'; 'Q'; 'R'; 'S'; 'T'; 'U'; 'V'; 'W'; 'X'; 'Y'; 'Z'; 'AA'; 'AB'; 'AC'; 'AD'; 'AE'; 'AF'; 'AG'; 'AH'; 'AI'};
AlphaTest = table(AlphaNum)
%%
table
WingLabel.AlphaNum = 1





%% Read Images

for i = G %1:length(WingIDS.Files) %1:length(WingIDS.Files)
		[test, fileinfo]= readimage(WingIDS,i);
		[filepath,name,ext] = fileparts(fileinfo.Filename);
		name = [name, ext];
		img(:,:,i) = rgb2gray(readimage(WingIDS,i));	
 	img1 = imgaussfilt(img(:,:,i), 4);
  img2 = imcomplement(img1);
  T = adaptthresh(img2,0.51,'ForegroundPolarity','Bright','NeighborhoodSize', 65, 'Statistic', 'Mean');
bw = imbinarize(img2, T);
% 		bw = imbinarize(img2, 'Adaptive', 'ForegroundPolarity', 'Bright', 'Sensitivity', 0.5);
		bw1 = bwareaopen(bw, 1000);
		bw2 = bwmorph(bw1, 'fill');
% 		WingIraw = img(:,:,i); 
%% Wing Identification
[N1,M1] = bwlabel(bw,4);
stats = regionprops(N1,'all');
WingArea = [stats.FilledArea];
[WingSize, WingID] = max(WingArea);
N1(find(N1~=WingID))=0;
NN = (N1~=0);
bw2 = bw.*NN;
BW = edge(bw2,'canny');
m0=find(sum(BW,1)>0);
n0=find(sum(BW,2)>0);
WingI1=imcrop(img(:,:,i),[min(m0),min(n0),max(m0)-min(m0),max(n0)-min(n0)]);
bw2=imcrop(bw2,[min(m0),min(n0),max(m0)-min(m0),max(n0)-min(n0)]);
% figure, imshow(bw2); %toggle2
bw3 = bwmorph(bw2, 'open', Inf);
bw4 = bwmorph(bw3, 'close', Inf);
bw5 = bwmorph(bw4, 'spur', Inf);
% figure, imshow(bw5); title('Smoothened; Extraneous Objects Removed')
bw5 = uint8(bw5);

% asdf = uint8(bw2);
% figure, imshow(WingI1.*asdf);
% ybwstore(floor(j/2)) = sum(sum(bw2))

%%
% 
% slopes = diff(ybwstore)./diff([1:50]);
%   figure, 
% %   plot([3:2:255], ybwstore); hold on;
%   plot([1:49], slopes); hold off;

  %%
  
WingConvex = [stats.ConvexArea];
WingConvexArea(i,1) = max(WingConvex);
%% Skeleton
bw6 = bwmorph(bw5,'thin', Inf); %Replaced skel
% bw6 = bwmorph(bw5,'thin');
bw7 = bwmorph(bw6,'spur', 11);%Spur setting is manually adjusted
se = strel('disk',1);
bw7a = imdilate(bw7,se);
% bw7 = bwmorph(bw7, 'thicken'); 
% figure, imshow(bw7); title('Skeleton')
bw8 = bwmorph(bw7, 'branchpoints');
asdf = regionprops(bw8,'all');
centroids = cat(1, asdf.Centroid);
% imshow(bw7); hold on;
% plot(centroids(:,1), centroids(:,2), 'g*')
% hold off
% bw8 = bwmorph(bw8, 'thicken', 3); %Discriminates against certain obj
se = strel('disk',5);
bw8 = imdilate(bw8,se);
bw9 = bwmorph(bw7, 'endpoints');
asdf2 = regionprops(bw9,'all');
centroids2 = cat(1, asdf2.Centroid);
% imshow(bw7)
% hold on
% plot(centroids2(:,1), centroids2(:,2), 'b*')
% hold off
bw9 = imdilate(bw9,se);
%%
[bw10, bw11] = find(bwmorph(bw7, 'branchpoints'));
[bw12, bw13] = find(bwmorph(bw7, 'endpoints'));
bw14{i} = [bw10, bw11];
bw15{i} = [bw12, bw13];
bw16{i} = [bw14{1,i}; bw15{1,i}]; %Set of all possible POIs
%%


%%
% figure
% for k = 1:268
% scatter(bw14{k}(:,2), bw14{k}(:,1), '.'); hold on;
% end
% set(gca, 'Ydir', 'reverse');
% ylim([0 1000]);
% box on;

%%
% [d,Za,Ta] = procrustes([bw14{1}(:,2) bw14{1}(:,1)],[bw14{2}(:,2) bw14{2}(:,1)])

bwCA = [stats.ConvexArea];
[bwCA2, bwCA4] = sort(bwCA, 'descend');
bwCA3 = bwCA4(1);
bwCA5 = stats(bwCA3).ConvexImage;
%% Skeletal Overlay
figure, 
% imshow(WingI1); title(name, 'Interpreter', 'none');
imshow(bwCA5); title(name, 'Interpreter', 'none');
I = bw7;
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
%% Step Clear
clear img1
clear img2
clear bw
clear bw1
end

%%
% T = adaptthresh(img2,0.5,'ForegroundPolarity','Bright','NeighborhoodSize', 35, 'Statistic', 'Mean');
% bw = imbinarize(img2, T);
% 		bw1 = bwareaopen(bw, 1000);
% 		bw2 = bwmorph(bw1, 'fill');
% [N1,M1] = bwlabel(bw,4);
% stats = regionprops(N1,'all');
% WingArea = [stats.FilledArea];
% [WingSize, WingID] = max(WingArea);
% N1(find(N1~=WingID))=0;
% NN = (N1~=0);
% bw2 = bw.*NN;
% BW = edge(bw2,'canny');
% m0=find(sum(BW,1)>0);
% n0=find(sum(BW,2)>0);
% WingI1=imcrop(img(:,:,i),[min(m0),min(n0),max(m0)-min(m0),max(n0)-min(n0)]);
% bw2=imcrop(bw2,[min(m0),min(n0),max(m0)-min(m0),max(n0)-min(n0)]);
% 
% % figure, imshow(bw2)
% 
% asdf = uint8(bw2);
% figure, imshow(WingI1.*asdf);

%%



%% Wing Size plot
% imgS = (stats(1).ConvexImage)
% figure,
% scatter(ones(1,length(A)), WingConvexArea(A), '.'); hold on;
% scatter(ones(1,length(B))*2, WingConvexArea(B), '.');
% scatter(ones(1,length(C))*3, WingConvexArea(C), '.');
% scatter(ones(1,length(D))*4, WingConvexArea(D), '.');
% scatter(ones(1,length(E))*5, WingConvexArea(E), '.');
% scatter(ones(1,length(F))*6, WingConvexArea(F), '.');
% scatter(ones(1,length(G))*7, WingConvexArea(G), '.');
% scatter(ones(1,length(H))*8, WingConvexArea(H), '.');
% scatter(ones(1,length(I))*9, WingConvexArea(I), '.');
% scatter(ones(1,length(J))*10, WingConvexArea(J), '.');
% scatter(ones(1,length(K))*11, WingConvexArea(K), '.');
% scatter(ones(1,length(L))*12, WingConvexArea(L), '.');
% scatter(ones(1,length(M))*13, WingConvexArea(M), '.');
% scatter(ones(1,length(N))*14, WingConvexArea(N), '.');
% scatter(ones(1,length(O))*15, WingConvexArea(O), '.');
% scatter(ones(1,length(P))*16, WingConvexArea(P), '.');
% scatter(ones(1,length(Q))*17, WingConvexArea(Q), '.');
% scatter(ones(1,length(R))*18, WingConvexArea(R), '.');
% scatter(ones(1,length(S))*19, WingConvexArea(S), '.');
% scatter(ones(1,length(T))*20, WingConvexArea(T), '.');
% scatter(ones(1,length(U))*21, WingConvexArea(U), '.');
% scatter(ones(1,length(V))*22, WingConvexArea(V), '.'); 
% scatter(ones(1,length(W))*23, WingConvexArea(W), '.'); 
% scatter(ones(1,length(X))*24, WingConvexArea(X), '.'); 
% scatter(ones(1,length(Y))*25, WingConvexArea(Y), '.'); 
% scatter(ones(1,length(Z))*26, WingConvexArea(Z), '.'); 
% scatter(ones(1,length(AA))*27, WingConvexArea(AA), '.'); 
% scatter(ones(1,length(AB))*28, WingConvexArea(AB), '.'); 
% scatter(ones(1,length(AC))*29, WingConvexArea(AC), '.'); 
% scatter(ones(1,length(AD))*30, WingConvexArea(AD), '.'); 
% scatter(ones(1,length(AE))*31, WingConvexArea(AE), '.'); 
% scatter(ones(1,length(AF))*32, WingConvexArea(AF), '.'); 
% scatter(ones(1,length(AG))*33, WingConvexArea(AG), '.'); 
% scatter(ones(1,length(AH))*34, WingConvexArea(AH), '.'); 
% scatter(ones(1,length(AI))*35, WingConvexArea(AI), '.'); 
% 
% scatter(0.7, mean(WingConvexArea(A)), '+k');
% scatter(1.7, mean(WingConvexArea(B)), '+k');
% scatter(2.7, mean(WingConvexArea(C)), '+k');
% scatter(3.7, mean(WingConvexArea(D)), '+k');
% scatter(4.7, mean(WingConvexArea(E)), '+k');
% scatter(5.7, mean(WingConvexArea(F)), '+k');
% scatter(6.7, mean(WingConvexArea(G)), '+k');
% scatter(7.7, mean(WingConvexArea(H)), '+k');
% scatter(8.7, mean(WingConvexArea(I)), '+k');
% scatter(9.7, mean(WingConvexArea(J)), '+k');
% scatter(10.7, mean(WingConvexArea(K)), '+k');
% scatter(11.7, mean(WingConvexArea(L)), '+k');
% scatter(12.7, mean(WingConvexArea(M)), '+k');
% scatter(13.7, mean(WingConvexArea(N)), '+k');
% scatter(14.7, mean(WingConvexArea(O)), '+k');
% scatter(15.7, mean(WingConvexArea(P)), '+k');
% scatter(16.7, mean(WingConvexArea(Q)), '+k');
% scatter(17.7, mean(WingConvexArea(R)), '+k');
% scatter(18.7, mean(WingConvexArea(S)), '+k');
% scatter(19.7, mean(WingConvexArea(T)), '+k');
% scatter(20.7, mean(WingConvexArea(U)), '+k');
% scatter(21.7, mean(WingConvexArea(V)), '+k'); 
% scatter(22.7, mean(WingConvexArea(W)), '+k'); 
% scatter(23.7, mean(WingConvexArea(X)), '+k'); 
% scatter(24.7, mean(WingConvexArea(Y)), '+k'); 
% scatter(25.7, mean(WingConvexArea(Z)), '+k'); 
% scatter(26.7, mean(WingConvexArea(AA)), '+k'); 
% scatter(27.7, mean(WingConvexArea(AB)), '+k'); 
% scatter(28.7, mean(WingConvexArea(AC)), '+k'); 
% scatter(29.7, mean(WingConvexArea(AD)), '+k'); 
% scatter(30.7, mean(WingConvexArea(AE)), '+k'); 
% scatter(31.7, mean(WingConvexArea(AF)), '+k'); 
% scatter(32.7, mean(WingConvexArea(AG)), '+k'); 
% scatter(33.7, mean(WingConvexArea(AH)), '+k'); 
% scatter(34.7, mean(WingConvexArea(AI)), '+k'); 
% 
% errorbar(0.7, mean(WingConvexArea(A)),std(WingConvexArea(A)), 'k');
% errorbar(1.7, mean(WingConvexArea(B)),std(WingConvexArea(B)), 'k');
% errorbar(2.7, mean(WingConvexArea(C)),std(WingConvexArea(C)), 'k');
% errorbar(3.7, mean(WingConvexArea(D)),std(WingConvexArea(D)), 'k');
% errorbar(4.7, mean(WingConvexArea(E)),std(WingConvexArea(E)), 'k');
% errorbar(5.7, mean(WingConvexArea(F)),std(WingConvexArea(F)), 'k');
% errorbar(6.7, mean(WingConvexArea(G)),std(WingConvexArea(G)), 'k');
% errorbar(7.7, mean(WingConvexArea(H)),std(WingConvexArea(H)), 'k');
% errorbar(8.7, mean(WingConvexArea(I)),std(WingConvexArea(I)), 'k');
% errorbar(9.7, mean(WingConvexArea(J)),std(WingConvexArea(J)), 'k');
% errorbar(10.7, mean(WingConvexArea(K)),std(WingConvexArea(K)), 'k');
% errorbar(11.7, mean(WingConvexArea(L)),std(WingConvexArea(L)), 'k');
% errorbar(12.7, mean(WingConvexArea(M)),std(WingConvexArea(M)), 'k');
% errorbar(13.7, mean(WingConvexArea(N)),std(WingConvexArea(N)), 'k');
% errorbar(14.7, mean(WingConvexArea(O)),std(WingConvexArea(O)), 'k');
% errorbar(15.7, mean(WingConvexArea(P)),std(WingConvexArea(P)), 'k');
% errorbar(16.7, mean(WingConvexArea(Q)),std(WingConvexArea(Q)), 'k');
% errorbar(17.7, mean(WingConvexArea(R)),std(WingConvexArea(R)), 'k');
% errorbar(18.7, mean(WingConvexArea(S)),std(WingConvexArea(S)), 'k');
% errorbar(19.7, mean(WingConvexArea(T)),std(WingConvexArea(T)), 'k');
% errorbar(20.7, mean(WingConvexArea(U)),std(WingConvexArea(U)), 'k');
% errorbar(21.7, mean(WingConvexArea(V)),std(WingConvexArea(V)), 'k');
% errorbar(22.7, mean(WingConvexArea(W)),std(WingConvexArea(W)), 'k');
% errorbar(23.7, mean(WingConvexArea(X)),std(WingConvexArea(X)), 'k');
% errorbar(24.7, mean(WingConvexArea(Y)),std(WingConvexArea(Y)), 'k');
% errorbar(25.7, mean(WingConvexArea(Z)),std(WingConvexArea(Z)), 'k');
% errorbar(26.7, mean(WingConvexArea(AA)),std(WingConvexArea(AA)), 'k');
% errorbar(27.7, mean(WingConvexArea(AB)),std(WingConvexArea(AB)), 'k');
% errorbar(28.7, mean(WingConvexArea(AC)),std(WingConvexArea(AC)), 'k');
% errorbar(29.7, mean(WingConvexArea(AD)),std(WingConvexArea(AD)), 'k');
% errorbar(30.7, mean(WingConvexArea(AE)),std(WingConvexArea(AE)), 'k');
% errorbar(31.7, mean(WingConvexArea(AF)),std(WingConvexArea(AF)), 'k');
% errorbar(32.7, mean(WingConvexArea(AG)),std(WingConvexArea(AG)), 'k');
% errorbar(33.7, mean(WingConvexArea(AH)),std(WingConvexArea(AH)), 'k');
% errorbar(34.7, mean(WingConvexArea(AI)),std(WingConvexArea(AI)), 'k'); 
% xx = [0 36];
% yy = [mean(WingConvexArea(AI)), mean(WingConvexArea(AI))];
% plot(xx, yy, '--r'); hold off;
% xlim([0 36])
% box on;
% ylabel('Wing Size (ConvexArea)');
% l = legend('12xSPS-GBE-lacZ 2 copies', '12xSPS-GBE-lacZ 2 copies_ Del_het Ser_het', '12xSPS-GBE-lacZ 2 copies_ H_het', '4x12xCSL-GBE-lacZxDl[9P]', ...
% '4x12xCSL-GBE-lacZxDl[revF10]Ser[RX82]', '4x12xCSL-GBE-lacZxH[1]', '4x12xCSL-GBE-lacZxYW', 'Del_het Ser_het', 'Dl[9P] x yw', 'GBE-48xSPS-lacZ 2 copies',...
% 'H1_Ago1', 'H1_Cdk8', 'H1_CycC', 'H1_Med12', 'H1_Med13', 'H1_YW', 'H_het', 'N1 x yw (FG)', 'N55e11 x yw (FG)', 'N55e11 x yw (YK)', 'N[55e11] x GBE-12xSPS-lacZ', ...
% 'N[55e11]+_cdk8[K185]+_female_rightwing', 'N[55e11]Xago[1]', 'N[55e11]Xcdk8[K185]', 'N[55e11]XcycC[Y5]', 'N[55e11]Xkto[T241]', 'N[55e11]Xskd[T613]', ...
% 'N[55e11]xDl[9P]', 'N[55e11]xDl[revF10]Ser[RX82]', 'N[55e11]xH[1]', 'N[55e11]xSu(H)[IB115]', 'cdk8 x yw', 'cycC x yw', 'kto x yw', 'yw', 'Location', 'SouthEastOutside');
% set(l, 'Interpreter', 'none')
%% Wing Table
% WingTable = WingLabel;
% WingAreaMean = [];
% WingAreaStd = [];
% 
% WingAreaMean(1,1) = mean(WingConvexArea(A));
% WingAreaMean(2,1) = mean(WingConvexArea(B));
% WingAreaMean(3,1) = mean(WingConvexArea(C));
% WingAreaMean(4,1) = mean(WingConvexArea(D));
% WingAreaMean(5,1) = mean(WingConvexArea(E));
% WingAreaMean(6,1) = mean(WingConvexArea(F));
% WingAreaMean(7,1) = mean(WingConvexArea(G));
% WingAreaMean(8,1) = mean(WingConvexArea(H));
% WingAreaMean(9,1) = mean(WingConvexArea(I));
% WingAreaMean(10,1) = mean(WingConvexArea(J));
% WingAreaMean(11,1) = mean(WingConvexArea(K));
% WingAreaMean(12,1) = mean(WingConvexArea(L));
% WingAreaMean(13,1) = mean(WingConvexArea(M));
% WingAreaMean(14,1) = mean(WingConvexArea(N));
% WingAreaMean(15,1) = mean(WingConvexArea(O));
% WingAreaMean(16,1) = mean(WingConvexArea(P));
% WingAreaMean(17,1) = mean(WingConvexArea(Q));
% WingAreaMean(18,1) = mean(WingConvexArea(R));
% WingAreaMean(19,1) = mean(WingConvexArea(S));
% WingAreaMean(20,1) = mean(WingConvexArea(T));
% WingAreaMean(21,1) = mean(WingConvexArea(U));
% WingAreaMean(22,1) = mean(WingConvexArea(V));
% WingAreaMean(23,1) = mean(WingConvexArea(W));
% WingAreaMean(24,1) = mean(WingConvexArea(X));
% WingAreaMean(25,1) = mean(WingConvexArea(Y));
% WingAreaMean(26,1) = mean(WingConvexArea(Z));
% WingAreaMean(27,1) = mean(WingConvexArea(AA));
% WingAreaMean(28,1) = mean(WingConvexArea(AB));
% WingAreaMean(29,1) = mean(WingConvexArea(AC));
% WingAreaMean(30,1) = mean(WingConvexArea(AD));
% WingAreaMean(31,1) = mean(WingConvexArea(AE));
% WingAreaMean(32,1) = mean(WingConvexArea(AF));
% WingAreaMean(33,1) = mean(WingConvexArea(AG));
% WingAreaMean(34,1) = mean(WingConvexArea(AH));
% WingAreaMean(35,1) = mean(WingConvexArea(AI));
% 
% WingAreaStd(1,1) = std(WingConvexArea(A));
% WingAreaStd(2,1) = std(WingConvexArea(B));
% WingAreaStd(3,1) = std(WingConvexArea(C));
% WingAreaStd(4,1) = std(WingConvexArea(D));
% WingAreaStd(5,1) = std(WingConvexArea(E));
% WingAreaStd(6,1) = std(WingConvexArea(F));
% WingAreaStd(7,1) = std(WingConvexArea(G));
% WingAreaStd(8,1) = std(WingConvexArea(H));
% WingAreaStd(9,1) = std(WingConvexArea(I));
% WingAreaStd(10,1) = std(WingConvexArea(J));
% WingAreaStd(11,1) = std(WingConvexArea(K));
% WingAreaStd(12,1) = std(WingConvexArea(L));
% WingAreaStd(13,1) = std(WingConvexArea(M));
% WingAreaStd(14,1) = std(WingConvexArea(N));
% WingAreaStd(15,1) = std(WingConvexArea(O));
% WingAreaStd(16,1) = std(WingConvexArea(P));
% WingAreaStd(17,1) = std(WingConvexArea(Q));
% WingAreaStd(18,1) = std(WingConvexArea(R));
% WingAreaStd(19,1) = std(WingConvexArea(S));
% WingAreaStd(20,1) = std(WingConvexArea(T));
% WingAreaStd(21,1) = std(WingConvexArea(U));
% WingAreaStd(22,1) = std(WingConvexArea(V));
% WingAreaStd(23,1) = std(WingConvexArea(W));
% WingAreaStd(24,1) = std(WingConvexArea(X));
% WingAreaStd(25,1) = std(WingConvexArea(Y));
% WingAreaStd(26,1) = std(WingConvexArea(Z));
% WingAreaStd(27,1) = std(WingConvexArea(AA));
% WingAreaStd(28,1) = std(WingConvexArea(AB));
% WingAreaStd(29,1) = std(WingConvexArea(AC));
% WingAreaStd(30,1) = std(WingConvexArea(AD));
% WingAreaStd(31,1) = std(WingConvexArea(AE));
% WingAreaStd(32,1) = std(WingConvexArea(AF));
% WingAreaStd(33,1) = std(WingConvexArea(AG));
% WingAreaStd(34,1) = std(WingConvexArea(AH));
% WingAreaStd(35,1) = std(WingConvexArea(AI));
% 
WingAreaMean = table(WingAreaMean);
WingAreaStd = table(WingAreaStd);
WingTable = [WingTable WingAreaMean WingAreaStd];

figure,
for i = 1:length(WingIDS.Files)
scatter(bw16{1,i}(:,1), bw16{1,i}(:,2), '.'); hold on;
end
set(gca, 'YDir', 'reverse')


% %% Procrustes
% [d,Z,tr] = procrustes(bw16{1,1},  bw16{1,2});









