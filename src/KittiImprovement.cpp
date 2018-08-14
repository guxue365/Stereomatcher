#include <iostream>
#include <vector>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

cv::Mat InterpolateKittiGT(const cv::Mat& rImage, bool bMean = true);
double getMeanNeighborValues(const cv::Mat& rImage, int row, int col, int boxsize);
double getMedianNeighborValues(const cv::Mat& rImage, int row, int col, int boxsize);


int main() {
	//VideoCapture oFilestreamGT("E:/dataset_kitti/data_scene_flow/training/disp_occ_0/%06d_10.png");
	VideoCapture oFilestreamGT("/home/jung/2018EntwicklungStereoalgorithmus/data/kitti/data_scene_flow/training/disp_occ_0/%06d_10.png");

	if (!oFilestreamGT.isOpened()) {
		cout << "Error opening files" << endl;
		return -1;
	}

	Mat oDisparityGT;
	Mat oGTInterpolatedMean;
	Mat oGTInterpolatedMedian;

	for (int iFrame = 0; ; ++iFrame) {
		oFilestreamGT >> oDisparityGT;
		if (oDisparityGT.empty())	break;

		cvtColor(oDisparityGT, oDisparityGT, CV_BGR2GRAY);

		oDisparityGT.convertTo(oDisparityGT, CV_8U, 1.0);


		oGTInterpolatedMean = InterpolateKittiGT(oDisparityGT, true);
		oGTInterpolatedMedian = InterpolateKittiGT(oDisparityGT, false);

		/*oDisparityGT *= 3;
		oGTInterpolatedMean *= 3;
		oGTInterpolatedMedian *= 3;

		applyColorMap(oDisparityGT, oDisparityGT, COLORMAP_JET);
		applyColorMap(oGTInterpolatedMean, oGTInterpolatedMean, COLORMAP_JET);
		applyColorMap(oGTInterpolatedMedian, oGTInterpolatedMedian, COLORMAP_JET);

		imshow("Disparity GT", oDisparityGT);
		imshow("GT Interpolated mean", oGTInterpolatedMean);
		imshow("GT Interpolated median", oGTInterpolatedMedian);*/

		char filename[100];
		sprintf(filename, "%6.6d_10.png", iFrame);
		string sFullFilename = "/home/jung/2018EntwicklungStereoalgorithmus/data/kitti/data_scene_flow/training/disp_custom/";
		sFullFilename += filename;
		
		oGTInterpolatedMedian.convertTo(oGTInterpolatedMedian, CV_16U, 255.0);
		imwrite(sFullFilename, oGTInterpolatedMedian);

		/*char c = (char)waitKey();
		if (c == 27) 	break;*/
	}

	return 0;
}

cv::Mat InterpolateKittiGT(const cv::Mat& rImage, bool bMean) {
	cv::Mat oResult;
	rImage.copyTo(oResult);

	for (int i = 0; i < rImage.rows; ++i) {
		for (int j = 0; j < rImage.cols; ++j) {
			if (rImage.at<uchar>(i, j) == 0) {
				double dInterpolation = 0.0;
				if (bMean) {
					dInterpolation = getMeanNeighborValues(rImage, i, j, 3);
				}
				else {
					dInterpolation = getMedianNeighborValues(rImage, i, j, 3);
				}

				if (dInterpolation > 255.0) {
					cout << "Warning: Interpolated value too large" << endl;
				}
				if (dInterpolation > 0.0) {
					oResult.at<uchar>(i, j) = (uchar)dInterpolation;
				}
			}
		}
	}

	return oResult;
}

double getMeanNeighborValues(const cv::Mat& rImage, int row, int col, int boxsize) {

	if (boxsize > 10)	return 0.0;


	double dValue = 0.0;
	double dValidPixels = 0.0;

	int iRange = (int)(boxsize / 2);

	for (int i = row - iRange; i <= row + iRange; ++i) {
		for (int j = col - iRange; j <= col + iRange; ++j) {
			if (i == j)	continue;
			if (i < 0 || j < 0 || i >= rImage.rows || j >= rImage.cols)	continue;
			
			uchar val = rImage.at<uchar>(i, j);
			if (val > 0) {
				dValue += (double)val;
				dValidPixels += 1.0;
			}
		}
	}

	if (dValidPixels <2.0) {
		return getMeanNeighborValues(rImage, row, col, boxsize + 2);
	}

	dValue /= dValidPixels;

	return dValue;
}

double getMedianNeighborValues(const cv::Mat& rImage, int row, int col, int boxsize) {

	if (boxsize > 10)	return 0.0;
	if (boxsize % 2 == 0) {
		cout << "Error: boxsize is even" << endl;
	}

	vector<uchar> aValues;

	int iRange = (int)(boxsize / 2);

	for (int i = row - iRange; i <= row + iRange; ++i) {
		for (int j = col - iRange; j <= col + iRange; ++j) {
			if (i == j)	continue;
			if (i < 0 || j < 0 || i >= rImage.rows || j >= rImage.cols)	continue;

			uchar val = rImage.at<uchar>(i, j);
			if (val > 0) {
				aValues.push_back(val);
			}
		}
	}

	if (aValues.size()<2) {
		return getMeanNeighborValues(rImage, row, col, boxsize + 2);
	}

	size_t size = aValues.size();
	sort(aValues.begin(), aValues.end());
	if (size % 2 == 0)
	{
		return (double)((aValues[size / 2 - 1] + aValues[size / 2]) / 2);
	}
	return (double)(aValues[size / 2]);
}
