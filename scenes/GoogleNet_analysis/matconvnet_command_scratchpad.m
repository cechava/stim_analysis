 % Download a pre-trained CNN from the web (needed once).
urlwrite(...
  'http://www.vlfeat.org/matconvnet/models/imagenet-googlenet-dag.mat', ...
  'imagenet-googlenet-dag.mat') ;
% load the pre-trained CNN
net = dagnn.DagNN.loadobj(load('imagenet-googlenet-dag.mat')) ;
net.mode = 'test' ;

% load and preprocess an image
im = imread('peppers.png') ;
im_ = single(im) ; % note: 0-255 range
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage) ;

% run the CNN
net.eval({'data', im_}) ;

% obtain the CNN otuput
scores = net.vars(net.getVarIndex('prob')).value ;
scores = squeeze(gather(scores)) ;

% show the classification results
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im) ;
title(sprintf('%s (%d), score %.3f',...
net.meta.classes.description{best}, best, bestScore)) ;


%setting options to keep activations of relevant layers
net.vars(net.getVarIndex('data')).precious = true;%original data
net.vars(net.getVarIndex('norm2')).precious = true;%layer 1, 192 units total
net.vars(net.getVarIndex('icp2_in')).precious = true;%layers 2, 256 units total
net.vars(net.getVarIndex('icp2_out')).precious = true;%layers 3, 480 units total
net.vars(net.getVarIndex('icp3_out')).precious = true;%layers 4, 512 units total
net.vars(net.getVarIndex('icp4_out')).precious = true;%layers 5, 512 units total
net.vars(net.getVarIndex('icp5_out')).precious = true;%layers 5, 512 units total
net.vars(net.getVarIndex('icp6_out')).precious = true;%layers 6, 528 units total
net.vars(net.getVarIndex('icp7_out')).precious = true;%layers 7, 832 units total
net.vars(net.getVarIndex('icp8_out')).precious = true;%layers 8, 832 units total








%getting layer 1 activations
[szx,szy,num_nits] = size(net.vars(net.getVarIndex('norm2')).value);

%getting single unit activation
net.vars(net.getVarIndex('norm2')).value(:,:,unit_idx)




% . For the
%  first layer, we analyzed the outputs of 100 filters of the second normalization unit. For layers 2
%  to 8, we analyzed the outputs of 100 filters of the concatenation units. For each analyzed unit,
%  we chose the 100 filters to have the closest RFs to the center of the image.

for i = 1:154
    disp(net.vars(i).name)
end

