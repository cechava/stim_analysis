% clear all
% clc
% 
% addpath('/Users/cesar/Documents/CoxLab/Repos/stim_analysis/scenes/GoogleNet_analysis/matconvnet-1.0-beta25')
% 
% % Setup MatConvNet.
% run matlab/vl_setupnn ;
% 
% % load the pre-trained CNN
% net = dagnn.DagNN.loadobj(load('imagenet-googlenet-dag.mat')) ;
% net.mode = 'test' ;

%setting options to keep activations of relevant layers
net.vars(net.getVarIndex('data')).precious = true;%original data
net.vars(net.getVarIndex('norm2')).precious = true;%layer 1, 192 units total
net.vars(net.getVarIndex('icp8_out')).precious = true;%layers 8, 832 units total


%for im_count = 1:55
im_count = 1;

% load an image
img_folder = '/Users/cesar/Documents/CoxLab/Repos/scene_stimuli/Small_interim';
im_name = strcat(int2str(im_count),'.png');
oim = double(imread(fullfile(img_folder,im_name)));
oim2 = padarray(oim,96);%pad to account for dimensions of creen

im0 = cat(3,oim2,oim2,oim2);

%figure;imshow(uint8(im0))


im1 = single(im0) ; % note: 0-255 range
im1 = imresize(im1, net.meta.normalization.imageSize(1:2)) ;
im1 = bsxfun(@minus, im1, net.meta.normalization.averageImage) ;


% run the CNN
net.eval({'data', im1}) ;

%getting layer1 acitivity
%all_layer_responses = net.vars(net.getVarIndex('norm2')).value;
all_layer_responses = net.vars(net.getVarIndex('icp8_out')).value;
[szx,szy,num_filters] = size(all_layer_responses);

%get patch in center of feature map for each filter
center = floor(szx/2);
half_pool_size = 2;
x0 = center-half_pool_size;
x1 = center+half_pool_size;
y0 = center-half_pool_size;
y1 = center+half_pool_size;

%figure;imagesc(all_layer_responses(x0:x1,y0:y1,1))

%store patch of feature map for each filter
img_activity = [];
img_activity_norm = [];
for fidx = 1:num_filters
    filter_patch = all_layer_responses(x0:x1,y0:y1,fidx);
	img_activity = cat(2,img_activity,filter_patch(:));
    img_activity_norm = cat(2,img_activity_norm,filter_patch(:)/max(filter_patch(:)));
end

