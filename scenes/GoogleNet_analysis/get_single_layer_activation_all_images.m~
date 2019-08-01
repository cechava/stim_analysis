% clear all
% clc
% 
% addpath('/Users/cesar/Documents/CoxLab/Repos/stim_analysis/scenes/GoogleNet_analysis/matconvnet-1.0-beta25')
% 
% % Setup MatConvNet.
% run matlab/vl_setupnn ;

% load the pre-trained CNN
% net = dagnn.DagNN.loadobj(load('imagenet-googlenet-dag.mat')) ;
% net.mode = 'test' ;

%setting options to keep activations of relevant layers
net.vars(net.getVarIndex('data')).precious = true;%original data
net.vars(net.getVarIndex('norm2')).precious = true;%layer 1, 192 units total
net.vars(net.getVarIndex('icp8_out')).precious = true;%layers 8, 832 units total

layer_activity_array = [];
layer_activity_norm_array = [];

for im_count = 1:165
    disp(im_count)

    % load an image
    if im_count <56
    img_folder = '/Users/cesar/Documents/CoxLab/Repos/scene_stimuli/Small_interim';
    im_name = strcat(int2str(im_count),'.png');
    elseif im_count >55 && im_count<111
        img_folder = '/Users/cesar/Documents/CoxLab/Repos/scene_stimuli/TexMatch/8x4_small';
        im_name = strcat(int2str(im_count-55),'.png');
    elseif im_count > 110
       img_folder = '/Users/cesar/Documents/CoxLab/Repos/scene_stimuli/MagMatch/8x4_small';
        im_name = strcat(int2str(im_count-110),'.png');
    end
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
    
     layer_activity_array = cat(3,layer_activity_array,img_activity);
     layer_activity_norm_array = cat(3,layer_activity_norm_array,img_activity_norm);
end

%find where max response across all images didn't fall within our central
%patch
max_activity_per_filter = squeeze(max(squeeze(max(layer_activity_norm_array)),[],2));
exclude_idxs = find(max_activity_per_filter ~=1);

include_idxs = 1:num_filters;
include_idxs(exclude_idxs) = [];

layer_activity_array = layer_activity_array(:,include_idxs,:);


h5create('All_Images_layer8_activity.h5', '/layer_activity', size(layer_activity_array))
h5write('All_Images_layer8_activity.h5', '/layer_activity', layer_activity_array)