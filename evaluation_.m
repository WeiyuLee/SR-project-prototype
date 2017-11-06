close all; clear all;

%% Set the data path
Input_data_dir = 'D:/Doc/ML/SR-project/input/';
Target_data_dir = 'D:/Doc/ML/SR-project/target/';

%% DON'T MODIFY ANYTHING BELOWS

% Get the file list in the target folder
Input_data_list = dir(Input_data_dir);
Target_data_list = dir(Target_data_dir);

Input_file_num = length(Input_data_list);
Target_file_num = length(Target_data_list);

if(Input_file_num ~= Target_file_num)
    fprintf('ERROR: file number did not match!\n');
    return;    
end

mean_PSNR = 0;
mean_SSIM = 0;
num_Img = 0;
shave = 0;

for i = 3:Input_file_num
    Input_file_name = [Input_data_dir Input_data_list(i).name];
    Target_file_name = [Target_data_dir Target_data_list(i).name];
    
    % Read images
    img_input = imread(Input_file_name);
    img_target = imread(Target_file_name);
    
    % Measure the size
    [h, w, c] = size(img_input);
    
    % Convert to YCbCr
    img_input = rgb2ycbcr(img_input);
    img_target = rgb2ycbcr(img_target);
    
    % Only use Y channel to calculate PSNR
    img_input = img_input(:, :, 1);
    img_target = img_target(:, :, 1);    

    % Shave the border
    img_input = img_input(1+shave:h-shave , 1+shave:w-shave, 1);
    img_target = img_target(1+shave:h-shave , 1+shave:w-shave, 1);    
    
    curr_psnr = psnr(img_input, img_target);
    curr_ssim = ssim(img_input, img_target);
    
    mean_PSNR = mean_PSNR + curr_psnr;
    mean_SSIM = mean_SSIM + curr_ssim;
    
    num_Img  = num_Img + 1;
    
    fprintf(['[' Input_file_name '] PSNR: ', num2str(curr_psnr), ' SSIM: ', num2str(curr_ssim), '\n']);    
end

mean_PSNR = mean_PSNR / num_Img;
mean_SSIM = mean_SSIM / num_Img;

fprintf(['Avg. PSNR: ', num2str(mean_PSNR), '\n']);
fprintf(['Avg. SSIM: ', num2str(mean_SSIM), '\n']);
