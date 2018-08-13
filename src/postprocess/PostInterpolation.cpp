#include <postprocess/PostInterpolation.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;

PostInterpolation::PostInterpolation() {

}

PostInterpolation::~PostInterpolation() {

}

cv::Mat PostInterpolation::Postprocess(const cv::Mat& rImage) {
	cv::Mat oResult = InterpolateKittiGT(rImage);

	return oResult;
}

cv::Mat PostInterpolation::InterpolateKittiGT(const cv::Mat& rImage) {
	cv::Mat oResult;
	rImage.copyTo(oResult);

	for (int i = 0; i < rImage.rows; ++i) {
		for (int j = 0; j < rImage.cols; ++j) {
			if (rImage.at<uchar>(i, j) == 0) {
				double dInterpolation = getMedianNeighborValues(rImage, i, j, 3);

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

double PostInterpolation::getMeanNeighborValues(const cv::Mat& rImage, int row, int col, int boxsize) {

	if (boxsize > 10)	return 0.0;
	if (boxsize % 2 == 0) {
		cout << "Error: boxsize is even" << endl;
	}

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

double PostInterpolation::getMedianNeighborValues(const cv::Mat& rImage, int row, int col, int boxsize) {

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