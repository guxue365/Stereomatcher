#include <iostream>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;

string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

void onMouse(int event, int x, int y, int, void*) {
	if (event != EVENT_LBUTTONDOWN)		return;

	cout << "Mouse Event at: " << x << " | " << y << endl;
}

cv::Mat ComputeOpenCVDisparity(const cv::Mat& rLeft, const cv::Mat& rRight, int iBoxSize = 7);
cv::Mat ComputeCustomDisparityGray(const cv::Mat& rLeft, const cv::Mat& rRight);
cv::Mat ComputeCustomDisparityColor(const cv::Mat& rLeft, const cv::Mat& rRight);

double ComputeMatchingCostGray(int iRow, int iColLeft, int iColRight, const cv::Mat& rLeft, const cv::Mat& rRight);
double ComputeMatchingCostColor(int iRow, int iColLeft, int iColRight, const cv::Mat& rLeft, const cv::Mat& rRight);

bool isValidMinimumStrict(double dValMin, int iIndexMin, const std::vector<double>& aValues);
bool isValidMinimumStrict2(double dValMin, int iIndexMin, const std::vector<double>& aValues);
bool isValidMinimumVar(double dValMin, int iIndexMin, const std::vector<double>& aValues);


int main()
{
	VideoCapture oFilestreamLeft("E:/dataset_kitti/data_scene_flow/training/image_2/%06d_10.png");
	VideoCapture oFilestreamRight("E:/dataset_kitti/data_scene_flow/training/image_3/%06d_10.png");
	VideoCapture oFilestreamGT("E:/dataset_kitti/data_scene_flow/training/disp_noc_1/%06d_10.png");

	/*VideoCapture oFilestreamLeft("/home/jung/2018EntwicklungStereoalgorithmus/data/kitty/data_scene_flow/training/image_2/%06d_10.png");
	VideoCapture oFilestreamRight("/home/jung/2018EntwicklungStereoalgorithmus/data/kitty/data_scene_flow/training/image_3/%06d_10.png");
	VideoCapture oFilestreamGT("/home/jung/2018EntwicklungStereoalgorithmus/data/kitty/data_scene_flow/training/disp_noc_1/%06d_10.png");*/

	if (!oFilestreamLeft.isOpened() || !oFilestreamRight.isOpened()) {
		cout << "Error opening files" << endl;
		return -1;
	}

	Mat oFrameLeftColor;
	Mat oFrameLeftGray;
	Mat oFrameRightColor;
	Mat oFrameRightGray;

	Mat oFrameLeftDivX;
	Mat oFrameRightDivX;
	Mat oFrameLeftDivY;
	Mat oFrameRightDivY;
	Mat oFrameLeftDiv;
	Mat oFrameRightDiv;

	Mat oFrameLeftCanny;
	Mat oFrameRightCanny;

	Mat oDisparityOpenCV;
	Mat oDisparityCustomGray;
	Mat oDisparityCustomColor;
	Mat oDisparityCustomDiv;
	Mat oDisparityCanny;
	Mat oDisparityGT;

	for(int iFrame=0; ; ++iFrame) {
		oFilestreamLeft >> oFrameLeftColor;
		oFilestreamRight >> oFrameRightColor;
		oFilestreamGT >> oDisparityGT;
		oDisparityGT.convertTo(oDisparityGT, CV_8U, 1.0 / 255.0);

		if (oFrameLeftColor.empty() || oFrameRightColor.empty() || oDisparityGT.empty())	break;
		
		cvtColor(oFrameLeftColor, oFrameLeftGray, COLOR_BGR2GRAY);
		cvtColor(oFrameRightColor, oFrameRightGray, COLOR_BGR2GRAY);

		Sobel(oFrameLeftGray, oFrameLeftDivX, CV_8U, 1, 0);
		Sobel(oFrameLeftGray, oFrameLeftDivY, CV_8U, 0, 1);

		Sobel(oFrameRightGray, oFrameRightDivX, CV_8U, 1, 0);
		Sobel(oFrameRightGray, oFrameRightDivY, CV_8U, 0, 1);

		Sobel(oFrameLeftGray, oFrameLeftDiv, CV_8U, 1, 1);
		Sobel(oFrameRightGray, oFrameRightDiv, CV_8U, 1, 1);

		Canny(oFrameLeftGray, oFrameLeftCanny, 100.0, 200.0);
		Canny(oFrameRightGray, oFrameRightCanny, 100.0, 200.0);

		oDisparityOpenCV = ComputeOpenCVDisparity(oFrameLeftGray, oFrameRightGray);

		oDisparityCustomGray = ComputeCustomDisparityGray(oFrameLeftGray, oFrameRightGray);
		oDisparityCustomColor = ComputeCustomDisparityColor(oFrameLeftColor, oFrameRightColor);
		oDisparityCustomDiv = ComputeOpenCVDisparity(oFrameLeftDiv, oFrameRightDiv, 21);
		oDisparityCanny = ComputeOpenCVDisparity(oFrameLeftCanny, oFrameRightCanny, 21);

		

		oDisparityOpenCV *= 3;
		oDisparityCustomGray *= 3;
		oDisparityCustomColor *= 3;
		oDisparityCustomDiv *= 3;
		oDisparityCanny *= 3;
		oDisparityGT *= 3;

		/*normalize(oDisparityOpenCV, oDisparityOpenCV, 0.0, 255.0, CV_MINMAX);
		normalize(oDisparityCustom, oDisparityCustom, 0.0, 255.0, CV_MINMAX);
		normalize(oDisparityGT, oDisparityGT, 0.0, 255.0, CV_MINMAX);*/

		applyColorMap(oDisparityOpenCV, oDisparityOpenCV, COLORMAP_JET);
		applyColorMap(oDisparityCustomGray, oDisparityCustomGray, COLORMAP_JET);
		applyColorMap(oDisparityCustomColor, oDisparityCustomColor, COLORMAP_JET);
		applyColorMap(oDisparityCustomDiv, oDisparityCustomDiv, COLORMAP_JET);
		applyColorMap(oDisparityCanny, oDisparityCanny, COLORMAP_JET);
		applyColorMap(oDisparityGT, oDisparityGT, COLORMAP_JET);

		

		imshow("Disparity OpenCV", oDisparityOpenCV);
		imshow("Disparity Custom Gray", oDisparityCustomGray);
		imshow("Disparity Custom Color", oDisparityCustomColor);
		imshow("Disparity Custom Div", oDisparityCustomDiv);
		imshow("Disparity Canny", oDisparityCanny);
		imshow("Disparity Ground Truth", oDisparityGT);

		imshow("Left Div X", oFrameLeftDivX);
		imshow("Left Div Y", oFrameLeftDivY);
		imshow("Left Div", oFrameLeftDiv);
		imshow("Left Canny", oFrameLeftCanny);

		setMouseCallback("Disparity OpenCV", onMouse);
		setMouseCallback("Disparity Custom Gray", onMouse);
		setMouseCallback("Disparity Custom Color", onMouse);

		

		imwrite("Color.png", oFrameLeftColor);
		imwrite("Gray.png", oFrameLeftGray);
		imwrite("Div.png", oFrameLeftDiv);
		imwrite("Canny.png", oFrameLeftCanny);
		imwrite("OpenCV.png", oDisparityOpenCV);
		imwrite("Custom_Gray.png", oDisparityCustomGray);
		imwrite("Custom_Color.png", oDisparityCustomColor);
		imwrite("Custom_Div.png", oDisparityCustomDiv);
		imwrite("Custom_Canny.png", oDisparityCanny);

		char c = (char)waitKey();
		if (c == 27) 	break;
	}

    return 0;
}

cv::Mat ComputeOpenCVDisparity(const cv::Mat& rLeft, const cv::Mat& rRight, int iBoxSize) {
	cv::Mat oResult;

	cv::Ptr<StereoBM> pStereoBM = StereoBM::create(0, iBoxSize);
	pStereoBM->compute(rLeft, rRight, oResult);

	oResult.convertTo(oResult, CV_8U, 1.0/16.0);

	return oResult;
}

cv::Mat ComputeCustomDisparityGray(const cv::Mat& rLeft, const cv::Mat& rRight) {
	assert(rLeft.rows == rRight.rows);
	assert(rLeft.cols == rRight.cols);
	assert(rLeft.type() == rRight.type());
	assert(rLeft.type() == CV_8U);

	int m = rLeft.rows;
	int n = rLeft.cols;

	const int iMaxDisparity = 128;

	cv::Mat oResult(m, n, CV_8U, Scalar(0));

	for (int i = 3; i < m-3; ++i) {
		cout << "Row " << i << " from " << m << endl;
		for (int j = iMaxDisparity; j < n- iMaxDisparity; ++j) {
			// match pixel rLeft(i, j) to any Pixel(i, *) on the right image
			// -> iterate through row i on the right image and compute cost
			
			vector<double> aMatchingCost(n);
			double dMin = std::numeric_limits<double>::max();
			int iCustomDisp = -1;
			vector<double> aDisp(iMaxDisparity);
			for (int k = j- iMaxDisparity+1; k < j+1; ++k) {
				//compute cost for pixel (i, j) and (i, k)
				double dCost = ComputeMatchingCostGray(i, j, k, rLeft, rRight);
				aDisp[j - k] = dCost;
				//cout << (int)(j - k) << endl;
				if (dCost < dMin) {
					dMin = dCost;
					iCustomDisp = j - k;
				}
			}

			auto itMin = std::min_element(aDisp.begin(), aDisp.end());
			int iMin = (int)std::distance(aDisp.begin(), itMin);
			double dMinVal = *itMin;
			
			if (isValidMinimumStrict(dMinVal, iMin, aDisp)) {
				oResult.at<uchar>(i, j) = (uchar)(iCustomDisp);
			}

		}
	}
	return oResult;
} 

cv::Mat ComputeCustomDisparityColor(const cv::Mat& rLeft, const cv::Mat& rRight) {
	assert(rLeft.rows == rRight.rows);
	assert(rLeft.cols == rRight.cols);
	assert(rLeft.type() == rRight.type());
	assert(rLeft.type() == CV_8UC3);

	int m = rLeft.rows;
	int n = rLeft.cols;

	const int iMaxDisparity = 128;

	cv::Mat oResult(m, n, CV_8U, Scalar(0));

	for (int i = 3; i < m - 3; ++i) {
		cout << "Row " << i << " from " << m << endl;
		for (int j = iMaxDisparity; j < n - iMaxDisparity; ++j) {
			// match pixel rLeft(i, j) to any Pixel(i, *) on the right image
			// -> iterate through row i on the right image and compute cost

			vector<double> aMatchingCost(n);
			double dMin = std::numeric_limits<double>::max();
			int iCustomDisp = -1;
			vector<double> aDisp(iMaxDisparity);
			for (int k = j - iMaxDisparity + 1; k < j + 1; ++k) {
				//compute cost for pixel (i, j) and (i, k)
				double dCost = ComputeMatchingCostColor(i, j, k, rLeft, rRight);
				aDisp[j - k] = dCost;
				//cout << (int)(j - k) << endl;
				if (dCost < dMin) {
					dMin = dCost;
					iCustomDisp = j - k;
				}
			}

			auto itMin = std::min_element(aDisp.begin(), aDisp.end());
			int iMin = (int)std::distance(aDisp.begin(), itMin);
			double dMinVal = *itMin;

			if (isValidMinimumStrict(dMinVal, iMin, aDisp)) {
				oResult.at<uchar>(i, j) = (uchar)(iCustomDisp);
			}

		}
	}
	return oResult;
}


double ComputeMatchingCostGray(int iRow, int iColLeft, int iColRight, const cv::Mat& rLeft, const cv::Mat& rRight) {
	assert(rLeft.type() == CV_8U);
	assert(rRight.type() == CV_8U);

	double dResult = 0.0;

	int iBoxSize = 7;
	for (int i = 0; i < iBoxSize; ++i) {
		int iCurrentRow = iRow + i - iBoxSize / 2;
		if (iCurrentRow < 0 || iCurrentRow >= rLeft.rows)	continue;

		for (int j = 0; j < iBoxSize; ++j) {
			int iCurrentColLeft = iColLeft + j - iBoxSize / 2;
			int iCurrentColRight = iColRight + j - iBoxSize / 2;
			if (iCurrentColLeft < 0 || iCurrentColRight<0 || iCurrentColLeft >= rLeft.cols || iCurrentColRight>rLeft.cols)	continue;

			double dLeft = (double)rLeft.at<uchar>((int)iCurrentRow, (int)iCurrentColLeft);
			double dRight = (double)rRight.at<uchar>((int)iCurrentRow, (int)iCurrentColRight);
			dResult += abs(dLeft - dRight);
		}
	}

	return dResult;
}

double ComputeMatchingCostColor(int iRow, int iColLeft, int iColRight, const cv::Mat& rLeft, const cv::Mat& rRight) {
	assert(rLeft.type() == CV_8UC3);
	assert(rRight.type() == CV_8UC3);

	double dResult = 0.0;

	int iBoxSize = 7;
	for (int i = 0; i < iBoxSize; ++i) {
		int iCurrentRow = iRow + i - iBoxSize / 2;
		if (iCurrentRow < 0 || iCurrentRow >= rLeft.rows)	continue;

		for (int j = 0; j < iBoxSize; ++j) {
			int iCurrentColLeft = iColLeft + j - iBoxSize / 2;
			int iCurrentColRight = iColRight + j - iBoxSize / 2;
			if (iCurrentColLeft < 0 || iCurrentColRight<0 || iCurrentColLeft >= rLeft.cols || iCurrentColRight>rLeft.cols)	continue;

			Vec3b dLeft = (Vec3b)rLeft.at<Vec3b>((int)iCurrentRow, (int)iCurrentColLeft);
			Vec3b dRight = (Vec3b)rRight.at<Vec3b>((int)iCurrentRow, (int)iCurrentColRight);
			double d11 = (double)dLeft[0];
			double d12 = (double)dLeft[1];
			double d13 = (double)dLeft[2];

			double d21 = (double)dRight[0];
			double d22 = (double)dRight[1];
			double d23 = (double)dRight[2];

			dResult += abs(d11 - d21) + abs(d12 - d22) + abs(d13 - d23);
		}
	}

	return dResult;
}

bool isValidMinimumStrict(double dValMin, int iIndexMin, const std::vector<double>& aValues) {
	double eps = 15.0;
	for (size_t l = 0; l < aValues.size(); ++l) {
		if (l == iIndexMin)		continue;
		if (abs(aValues[l] - dValMin) < eps) {
			return false;
		}
	}
	return true;
}

bool isValidMinimumVar(double dValMin, int iIndexMin, const std::vector<double>& aValues) {
	int n = 10;

	vector<double> aCopy(aValues);
	aCopy[iIndexMin] = std::numeric_limits<double>::max();

	sort(aCopy.begin(), aCopy.end());
	aCopy.resize(n);

	/*cout << "After sort: " << endl;
	for (auto& dVal : aCopy) {
		cout << dVal << endl;
	}*/

	double dMean = 0.0;
	for (auto& dVal : aCopy) {
		dMean += dVal;
	}
	dMean /= (double)(aCopy.size());

	double dVar = 0.0;
	for (auto& dVal : aCopy) {
		dVar += (dVal - dMean)*(dVal - dMean);
	}
	dVar = sqrt(dVar / (double)(aCopy.size() - 1));

	/*cout << "Mean: " << dMean << endl;
	cout << "Var: " << dVar << endl;
	cout << "Val: " << dValMin << endl;*/

	if (dValMin+sqrt(dVar) < aCopy[0]) {
		return true;
	}

	return false;
}

bool isValidMinimumStrict2(double dValMin, int iIndexMin, const std::vector<double>& aValues) {
	vector<double> aCopy(aValues);

	double eps = 0.01;

	sort(aCopy.begin(), aCopy.end());

	if (abs((aCopy[0] - aCopy[1]) / aCopy[0]) < eps) {
		return false;
	}
	return true;
}
