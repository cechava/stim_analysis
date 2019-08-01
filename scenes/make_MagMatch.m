clear all
clc

addpath('/Users/cesar/Documents/CoxLab/imageProcessingTools/matlabPyrTools')


%im_count = 1;

for im_count = 55
    % load original image
    img_folder = '/Users/cesar/Documents/CoxLab/Repos/scene_stimuli/Small_interim';
    im_name = strcat(int2str(im_count),'.png');
    oim = double(imread(fullfile(img_folder,im_name)))';

    % set options
    out_folder = strcat('/Users/cesar/Documents/CoxLab/Repos/scene_stimuli/MagMatch_no/',int2str(im_count));
    mkdir(out_folder)
    output_arg = strcat('outputPath=',out_folder);
    opts = metamerOpts(oim,'windowType=square','nSquares=[1 1]',output_arg,'verbose=1');

    % make windows
    m = mkImMasks(opts);

    % plot windows
    %plotWindows(m,opts);

    % do metamer analysis on original (measure statistics)
    params = metamerAnalysis(oim,m,opts);

    % do metamer synthesis (generate new image matched for statistics)
    res = metamerSynthesis_noPix(params,size(oim),m,opts);

end