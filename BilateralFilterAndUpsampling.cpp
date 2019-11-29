#include<opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include<iostream>

using namespace std;
using namespace cv;

// Gaussian function
float gaussian(float rSquare, double sigma) {
	return exp(-rSquare / (2 * CV_PI * pow(sigma, 2)));
}

// Creating static spatial kernel (Gaussian)
Mat generateSpatialKernel(int kernelSize, double sigma) {
	int halfKernelSize = kernelSize / 2; 

	Mat spatialKernel = Mat::zeros(kernelSize, kernelSize, CV_32FC3);
	for (int x = 0; x < kernelSize; ++x) {
		for (int y = 0; y < kernelSize; ++y) {
			// Calculating Gaussian (same for all channels)
			float kernelElement = gaussian(pow(x - halfKernelSize, 2) + pow(y - halfKernelSize, 2), sigma);
			spatialKernel.at<Vec3f>(y, x) = Vec3f(kernelElement, kernelElement, kernelElement);
		}
	}

	return spatialKernel;
}

// Normalizing multichannel kernel
Mat normalizeKernel(Mat kernel) {
	// Splitting kernel into channels
	Mat kernelChannels[3];
	split(kernel, kernelChannels);

	// Normalizing kernel channels
	kernelChannels[0] /= cv::sum(kernelChannels[0])[0];
	kernelChannels[1] /= cv::sum(kernelChannels[1])[0];
	kernelChannels[2] /= cv::sum(kernelChannels[2])[0];

	// Merging kernel channels after normalization
	vector<Mat> channelsForMerge;
	channelsForMerge.push_back(kernelChannels[0]);
	channelsForMerge.push_back(kernelChannels[1]);
	channelsForMerge.push_back(kernelChannels[2]);
	Mat kernelMerged;
	merge(channelsForMerge, kernelMerged);

	return kernelMerged;
}

// Guided bilateral filter implementation (if inputImage is the same as the guideImage, it works as a simple bilateral filter)
Mat bilateralFilter(Mat inputImage, Mat guideImage, int kernelSize, double sigmaSpatial, double sigmaSpectral) {
	Mat outputImage = Mat::zeros(inputImage.size(), CV_32FC3);
	int halfKernelSize = kernelSize / 2;

	// Adding padding to input and guide images for being able to do convolution on the whole image
	Mat inputImagePadded = Mat::zeros(inputImage.rows + 2 * halfKernelSize, inputImage.cols + 2 * halfKernelSize, CV_32FC3);
	copyMakeBorder(inputImage, inputImagePadded, halfKernelSize, halfKernelSize, halfKernelSize, halfKernelSize, BORDER_CONSTANT, Scalar(0, 0, 0));
	Mat guideImagePadded = Mat::zeros(guideImage.rows + 2 * halfKernelSize, guideImage.cols + 2 * halfKernelSize, CV_32FC3);
	copyMakeBorder(guideImage, guideImagePadded, halfKernelSize, halfKernelSize, halfKernelSize, halfKernelSize, BORDER_CONSTANT, Scalar(0, 0, 0));

	// Creating static spatial kernel (Gaussian)
	Mat spatialKernel = generateSpatialKernel(kernelSize, sigmaSpatial);

	// Creating spectral kernels and doing convolution
	for (int i = halfKernelSize; i < inputImagePadded.cols - halfKernelSize; ++i) {
		for (int j = halfKernelSize; j < inputImagePadded.rows - halfKernelSize; ++j) {
			Vec3f centralIntensities = guideImagePadded.at<Vec3f>(j, i);

			// Creating spectral kernel
			Mat spectralKernel = Mat::zeros(kernelSize, kernelSize, CV_32FC3);
			for (int x = 0; x < kernelSize; ++x) {
				for (int y = 0; y < kernelSize; ++y) {
					Vec3f intensities = guideImagePadded.at<Vec3f>(j - halfKernelSize + y, i - halfKernelSize + x);

					// Calculating Gaussians (different for each channel)
					Vec3f kernelElements;
					kernelElements[0] = gaussian(pow(centralIntensities[0] - intensities[0], 2), sigmaSpectral);
					kernelElements[1] = gaussian(pow(centralIntensities[1] - intensities[1], 2), sigmaSpectral);
					kernelElements[2] = gaussian(pow(centralIntensities[2] - intensities[2], 2), sigmaSpectral);
					
					spectralKernel.at<Vec3f>(y, x) = kernelElements;
				}
			}

			// Creating kernel
			Mat kernel = spatialKernel.mul(spectralKernel);
			kernel = normalizeKernel(kernel);

			// Cutting window from image
			Rect ROI = Rect(i - halfKernelSize, j - halfKernelSize, kernelSize, kernelSize);
			Mat window = inputImagePadded(ROI);

			// Filtering
			Scalar filteredValue = cv::sum(window.mul(kernel));
			outputImage.at<Vec3f>(j - halfKernelSize, i - halfKernelSize)[0] = filteredValue[0];
			outputImage.at<Vec3f>(j - halfKernelSize, i - halfKernelSize)[1] = filteredValue[1];
			outputImage.at<Vec3f>(j - halfKernelSize, i - halfKernelSize)[2] = filteredValue[2];
		}
	}	
	return outputImage;
}

// Image upsampling using guided bilateral filter
Mat upsample(Mat inputImage, Mat guideImage, int kernelSize, double sigmaSpatial, double sigmaSpectral) {
	int upsampleFactor = log2(guideImage.rows / inputImage.rows);
	Mat upsampledImage = inputImage;

	for (int i = 1; i < upsampleFactor; ++i) {
		// Doubling the size of the image
		resize(upsampledImage, upsampledImage, cv::Size(), 2, 2);

		// Downscaling guide image
		Mat downscaledGuideImage;
		resize(guideImage, downscaledGuideImage, upsampledImage.size());

		// Filtering upscaled image
		upsampledImage = bilateralFilter(upsampledImage, downscaledGuideImage, kernelSize, sigmaSpatial, sigmaSpectral);
	}
	resize(upsampledImage, upsampledImage, guideImage.size());
	upsampledImage = bilateralFilter(upsampledImage, guideImage, kernelSize, sigmaSpatial, sigmaSpectral);

	return upsampledImage;
}

int main(int argc, char** argv)
{
	if (argc != 8) {
		cout << "7 command line arguments needed: Mode, Input image path, Output image path, Kernel size, Sigma for spatial filter, Sigma for spectral filter, Guide image path" << endl;
		return -1;
	}

	// Reading arguments
	String mode = argv[1];
	String inputImagePath = argv[2];
	String outputImagePath = argv[3];
	int kernelSize = atoi(argv[4]);
	double sigmaSpatial = atof(argv[5]);
	double sigmaSpectral = atof(argv[6]);
	String guideImagePath = argv[7];

	// Opening input image and converting to float
	Mat inputImage = imread(inputImagePath, CV_LOAD_IMAGE_COLOR);
	Mat inputImage_float;
	inputImage.convertTo(inputImage_float, CV_32FC3, 1.0 / 255.0);
	Mat outputImage;

	// Filtering or upsampling
	if (mode == "filtering") {
		outputImage = bilateralFilter(inputImage_float, inputImage_float, kernelSize, sigmaSpatial, sigmaSpectral);
	}
	else {
		resize(inputImage, inputImage, cv::Size(), 0.125, 0.125);
		Mat guideImage = imread(guideImagePath, CV_LOAD_IMAGE_COLOR);
		Mat guideImage_float;
		guideImage.convertTo(guideImage_float, CV_32FC3, 1.0 / 255.0);
		outputImage = upsample(inputImage_float, guideImage_float, kernelSize, sigmaSpatial, sigmaSpectral);
	}

	// Converting output image
	outputImage.convertTo(outputImage, CV_8UC3, 255.0);

	// Saving result
	imwrite(outputImagePath, outputImage);

	waitKey(0);
}