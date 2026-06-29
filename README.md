# Conditional DDPM for CBCT-to-Synthetic CT (sCT) Brain Image Synthesis

A PyTorch implementation and extension of a conditional Denoising Diffusion
Probabilistic Model (DDPM) that generates synthetic CT (sCT) from cone-beam CT
(CBCT) brain scans. The model operates on 2D axial slices: each noised CT slice
is conditioned on the corresponding CBCT slice (concatenated as an input channel
at every diffusion step) and the network is trained to denoise it, following
Peng et al.

Outputs are evaluated against the ground-truth CT with MAE (HU), PSNR,
RMSE, and SSIM, using the metric convention of Liang et al. (2019):
DATA_RANGE = 4071, PSNR = 20 * log10(4071 / RMSE).

## Requirements
- Python 3.9+
- PyTorch (CUDA-enabled; trained on a single RTX 4090)
- numpy, pydicom, scikit-image, tqdm, matplotlib, nibabel

## Citation
If you use this code, please cite the original method:

@article{peng2024cbct,
  title   = {{CBCT}-based synthetic {CT} image generation using conditional denoising diffusion probabilistic model},
  author  = {Peng, Junbo and Qiu, Richard L. J. and Wynne, Jacob F. and Chang, Chih-Wei and Pan, Shaoyan and Wang, Tonghe and Roper, Justin and Liu, Tian and Patel, Pretesh R. and Yu, David S. and Yang, Xiaofeng},
  journal = {Medical Physics},
  year    = {2024},
  doi     = {10.1002/mp.16704}
}