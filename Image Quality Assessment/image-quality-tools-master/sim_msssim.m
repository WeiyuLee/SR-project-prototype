function [MSSSIM_Value] = sim_ssim(num_images)
    % if the num_images variable is not specified use the default value of
    % 1
    if ~exist('num_images','var')
        num_images = 1;
    end 
    
    currDir = cd ('images');
    [threshold] = textread('threshold.data', '%s');
    n = 100;
    MSSSIM_Value = zeros(n, 5);

    for i = 1:num_images
        src = strcat('image',int2str(i),'*Orig.jpg');
        origFile = dir(src);
        for j =  1:length(threshold)
            copy = strcat('image',int2str(i),'*', threshold{j}, 'Copy.jpg');
            fileNames = dir(copy);
             try
                origFileName = imread(origFile(1).name);
            catch exception
                error('the copy of image file doesnot exist');
            end
            
            filename = fileNames.name;
            
            try
                copyFileName = imread(filename);
            catch exception
                error('the copy of image file doesnot exist');
            end
            
            MSSSIM_Value(i, j) = msssim(rgb2gray(origFileName), rgb2gray(copyFileName));
        end
    end
    MSSSIM_Value = MSSSIM_Value(1:i, :);
    % go to src directory %
    cd ..;
end