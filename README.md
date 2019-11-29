# 3DSSF_BilateralFilterAndUpsampling
Project for 3D Sensing and Sensor Fusion subject at ELTE.

## Tasks
1. Implementing a bilateral filter. Evaluating it for 16 different combinations of 4-4 sigma values.
2. Converting the bilateral filter to a guided bilateral filter for guided image upsampling.

## Program parameters
| ID | Variable name | Value |
| --- | --- | --- |
| 1 | mode | Running filtering or upsampling. Values: filtering or upsampling |
| 2 | inputImagePath | Path of input image |
| 3 | outputImagePath | Path of output image |
| 4 | kernelSize | Size of kernel |
| 5 | sigmaSpatial | Sigma for spatial filtering |
| 6 | sigmaSpectral | Sigma for spectral filtering |
| 7 | guideImagePath | Path of guide image used for upsampling |

## Dataset
Face image with freckles for filtering. Source: https://www.pexels.com/photo/close-up-photography-of-woman-s-face-with-freckles-2709386

Reindeer depth and guide image for upsampling. Source: http://vision.middlebury.edu/stereo/data/scenes2005

## Evaluation
I tested the code for 16 different combinations of sigmas. The following table summarizes the results.

| Spatial sigma | Spectral sigma | Output path |
| --- | --- | --- |
| 0.1 | 0.1 | Output/Filtering/freckles_spat0dot1_spec0dot1.jpg |
| 0.1 | 1 | Output/Filtering/freckles_spat0dot1_spec1.jpg |
| 0.1 | 10 | Output/Filtering/freckles_spat0dot1_spec10.jpg |
| 0.1 | 100 | Output/Filtering/freckles_spat0dot1_spec100.jpg |
| 1 | 0.1 | Output/Filtering/freckles_spat1_spec0dot1.jpg |
| 1 | 1 | Output/Filtering/freckles_spat1_spec1.jpg |
| 1 | 10 | Output/Filtering/freckles_spat1_spec10.jpg |
| 1 | 100 | Output/Filtering/freckles_spat1_spec100.jpg |
| 10 | 0.1 | Output/Filtering/freckles_spat10_spec0dot1.jpg |
| 10 | 1 | Output/Filtering/freckles_spat10_spec1.jpg |
| 10 | 10 | Output/Filtering/freckles_spat10_spec10.jpg |
| 10 | 100 | Output/Filtering/freckles_spat10_spec100.jpg |
| 100 | 0.1 | Output/Filtering/freckles_spat100_spec0dot1.jpg |
| 100 | 1 | Output/Filtering/freckles_spat100_spec1.jpg |
| 100 | 10 | Output/Filtering/freckles_spat100_spec10.jpg |
| 100 | 100 | Output/Filtering/freckles_spat100_spec100.jpg |
