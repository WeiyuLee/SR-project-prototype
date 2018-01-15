target_path = 'EDSR_x2_torch_baseline/target/img_002_SRF_2_HR.png';
dis_path = 'EDSR_x2_torch_baseline/test/img_002_SRF_2_LR.png';
target_img = imread(target_path);
dis_img = imread(dis_path);

essim = ESSIM(target_img, dis_img)
mssim_score = msssim(rgb2gray(target_img), rgb2gray(dis_img))
ifc_score = metrix_mux(rgb2gray(target_img), rgb2gray(dis_img), 'VSNR')

target_path = 'EDSR_WGAN_v5_PatchWGAN/target/target_3.png';
dis_path = 'EDSR_WGAN_v5_PatchWGAN/test/test_3.png';
target_img = imread(target_path);
dis_img = imread(dis_path);

essim = ESSIM(target_img, dis_img)
mssim_score = msssim(rgb2gray(target_img), rgb2gray(dis_img))
ifc_score = metrix_mux(rgb2gray(target_img), rgb2gray(dis_img), 'VSNR')