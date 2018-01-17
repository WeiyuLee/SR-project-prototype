close all; clear all;

%% Set the data path
Input_data_dir = 'Y:/SR_project_evaluation/grid_eval_test/full_size/test/';
Target_data_dir = 'Y:/SR_project_evaluation/grid_eval_test/full_size/target/';

%Input_data_dir = 'Y:/SR_project_evaluation/EDSR_x2_torch_baseline/test/';%
%Target_data_dir = 'Y:/SR_project_evaluation/EDSR_x2_torch_baseline/target/';


%Input_data_dir = 'Y:/SR_project_evaluation/EDSR_WGAN_v5_PatchWGAN/test/';
%Target_data_dir = 'Y:/SR_project_evaluation/EDSR_WGAN_v5_PatchWGAN/target/';

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
mean_VSNR = 0;
mean_vif = 0;
mean_uqi = 0 ;
mean_vifp = 0 ;
mean_wsnr = 0;
mean_nqm = 0;
mean_mssim = 0;
mean_ifc = 0;
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
    curr_vsnr = metrix_mux(img_target,img_input , 'VSNR');
    curr_vif = metrix_mux(img_target,img_input , 'VIF');
    curr_uqi = metrix_mux(img_target,img_input , 'UQI');
    curr_vifp = metrix_mux(img_target,img_input , 'VIFP');
    curr_wsnr =  metrix_mux(img_target,img_input , 'WSNR');
    curr_nqm =  metrix_mux(img_target,img_input , 'NQM');
    curr_mssim =  metrix_mux(img_target,img_input , 'MSSIM');
    curr_ifc =  metrix_mux(img_target,img_input , 'IFC');
% mean-squared-error	      		    'MSE'
% peak signal-to-noise-ratio	      'PSNR'
% structural similarity index	      'SSIM' 
% multi-scale SSIM index	      		'MSSIM'-
% visual signal-to-noise ratio	    'VSNR'-
% visual information fidelity	      'VIF'-
% pixel-based VIF	      			      'VIFP -'
% universal quality index	      		'UQI'-
% information fidelity criterion	  'IFC'-
% noise quality measure	      		  'NQM'-
% weighted signal-to-noise ratio	  'WSNR'-
% signal-to-noise ratio	      		  'SNR'

    mean_PSNR = mean_PSNR + curr_psnr;
    mean_SSIM = mean_SSIM + curr_ssim;
    mean_VSNR = mean_VSNR + curr_vsnr;
    mean_vif = mean_vif + curr_vif;
    mean_uqi = mean_uqi + curr_uqi;
    mean_vifp = mean_vifp + curr_vifp;
    mean_wsnr = mean_wsnr + curr_wsnr;
    mean_nqm = mean_nqm + curr_nqm;
    mean_mssim = mean_mssim + curr_mssim;
    mean_ifc = mean_ifc + curr_ifc;
    num_Img  = num_Img + 1;
    
    fprintf(['[' Input_file_name '] PSNR: ', num2str(curr_psnr), ' SSIM: ', num2str(curr_ssim), '\n']);
    fprintf(['[' Input_file_name '] VSNR: ', num2str(curr_vsnr), ' VIF: ', num2str(curr_vif), '\n']);
    fprintf(['[' Input_file_name '] UQI: ', num2str(curr_uqi), ' VIFP: ', num2str(curr_vifp), '\n']);
    fprintf(['[' Input_file_name '] WSNR: ', num2str(curr_wsnr), ' NQM: ', num2str(curr_nqm), '\n']);
    fprintf(['[' Input_file_name '] MSSIM: ', num2str(curr_mssim), ' IFC: ', num2str(mean_ifc), '\n']);


end

mean_PSNR = mean_PSNR / num_Img;
mean_SSIM = mean_SSIM / num_Img;
mean_VSNR = mean_VSNR/num_Img;
mean_uqi = mean_uqi/num_Img;
mean_vif = mean_vif/num_Img;
mean_vifp = mean_vifp/num_Img;
mean_wsnr = mean_wsnr/num_Img;
mean_mssim = mean_mssim/num_Img;
mean_ifc = mean_ifc/num_Img;


fprintf(['Avg. PSNR: ', num2str(mean_PSNR), '\n']);
fprintf(['Avg. SSIM: ', num2str(mean_SSIM), '\n']);
fprintf(['Avg. VSNR: ', num2str(mean_VSNR), '\n']);
fprintf(['Avg. VIF: ', num2str(mean_vif), '\n']);
fprintf(['Avg. UQI: ', num2str(mean_uqi), '\n']);
fprintf(['Avg. VIFP: ', num2str(mean_vifp), '\n']);
fprintf(['Avg. WSNR: ', num2str(mean_wsnr), '\n']);
fprintf(['Avg. NQM: ', num2str(curr_nqm), '\n']);
fprintf(['Avg. MSSIM: ', num2str(mean_mssim), '\n']);
fprintf(['Avg. IFC: ', num2str(mean_ifc), '\n']);
