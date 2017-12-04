%% Clear
clear all; close all; clc;
%% Loading
run CoordinateArray.m %Manually curated coordinates
WingIDS = imageDatastore('C:\Users\che7oz\Desktop\Wing Quant\Wing Images\','FileExtensions','.tif', 'LabelSource', 'foldernames','IncludeSubfolders', 1)
%%




%% Load Images
WingFiles = dir('pwd\*.tif');
for i=1:length(WingFiles);

end
pth = genpath(pwd);
pth = strsplit(pth, ';')' % Separate into individual folders.
%%  Refine Below Example Code

dirinfo = dir();
dirinfo(~[dirinfo.isdir]) = [];  %remove non-directories
subdirinfo = cell(length(dirinfo));
for K = 1 : length(dirinfo)
  thisdir = dirinfo(K).name;
  subdirinfo{K} = dir(fullfile(thisdir, '*.tif'));
end
%%
% Start with a folder and get a list of all subfolders.
% Finds and prints names of all files in 
% that folder and all of its subfolders.
% Similar to imageSet() function in the Computer Vision System Toolbox: http://www.mathworks.com/help/vision/ref/imageset-class.html
clc;    % Clear the command window.
workspace;  % Make sure the workspace panel is showing.
format long g;
format compact;

% Define a starting folder.
start_path = fullfile(matlabroot, '\toolbox');
if ~exist(start_path, 'dir')
	start_path = matlabroot;
end
% Ask user to confirm or change.
uiwait(msgbox('Pick a starting folder on the next window that will come up.'));
topLevelFolder = uigetdir(start_path);
if topLevelFolder == 0
	return;
end
% Get list of all subfolders.
allSubFolders = genpath(topLevelFolder);
% Parse into a cell array.
remain = allSubFolders;
listOfFolderNames = {};
while true
	[singleSubFolder, remain] = strtok(remain, ';');
	if isempty(singleSubFolder)
		break;
	end
	listOfFolderNames = [listOfFolderNames singleSubFolder];
end
numberOfFolders = length(listOfFolderNames)

% Process all image files in those folders.
for k = 1 : numberOfFolders
	% Get this folder and print it out.
	thisFolder = listOfFolderNames{k};
	fprintf('Processing folder %s\n', thisFolder);
	
	% Get ALL files.
	filePattern = sprintf('%s/*.*', thisFolder);
	baseFileNames = dir(filePattern);
% 	% Get m files.
% 	filePattern = sprintf('%s/*.m', thisFolder);
% 	baseFileNames = dir(filePattern);
% 	% Add on FIG files.
% 	filePattern = sprintf('%s/*.fig', thisFolder);
% 	baseFileNames = [baseFileNames; dir(filePattern)];
	% Now we have a list of all files in this folder.
	
	numberOfImageFiles = length(baseFileNames);
	if numberOfImageFiles >= 1
		% Go through all those files.
		for f = 1 : numberOfImageFiles
			fullFileName = fullfile(thisFolder, baseFileNames(f).name);
			fprintf('     Processing file %s\n', fullFileName);
		end
	else
		fprintf('     Folder %s has no files in it.\n', thisFolder);
	end
end
%%


