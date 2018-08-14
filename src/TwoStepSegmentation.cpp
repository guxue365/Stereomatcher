#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <segmentation/DBSCAN.h>
#include <segmentation/RegionGrowing.h>

using namespace std;
using namespace cv;


std::vector<cv::Rect2i> ExtractRegions(const cv::Mat& rCoarseRegions, int iNumRegions);
std::vector<cv::Rect2i> MergeRegions(const std::vector<cv::Rect2i>& aRegions);

int main() {
	RegionGrowing oSegment;
	DBSCAN oDBScan;

	oDBScan.setEps(1.0);
	oDBScan.setMinPts(5);

	Mat oImage = imread("/home/jung/2018EntwicklungStereoalgorithmus/data/region_example2.png", CV_LOAD_IMAGE_GRAYSCALE);

	cout << "Image Size: " << oImage.size() << endl;

	Mat oImageTwoStep = Mat::zeros(oImage.size(), CV_8U);
	
	Mat oImageCoarse;
	double dScaling = 0.025;
	resize(oImage, oImageCoarse, Size(), dScaling, dScaling, INTER_LINEAR);
	threshold(oImageCoarse, oImageCoarse, 1.0, 255.0, THRESH_BINARY);
	threshold(oImage, oImage, 1.0, 255.0, THRESH_BINARY);

	
	Mat oRegionCoarse = oSegment.Segment(oImageCoarse);

	vector<Rect2i> aRegionsRaw = ExtractRegions(oRegionCoarse, 4);
	for (int i = 0; i < aRegionsRaw.size(); ++i) {
		cout << "Region " << i << ": " << endl;
		cout << aRegionsRaw[i] << endl;
	}

	cout << "Merged Regions: " << endl;
	vector<Rect2i> aRegions = MergeRegions(aRegionsRaw);
	for (int i = 0; i < aRegions.size(); ++i) {
		cout << "Region " << i << ": " << endl;
		cout << aRegions[i] << endl;
	}

	vector<Mat> aMatRegions(aRegions.size());
	vector<Mat> aDBScan(aRegions.size());
	for (size_t i = 0; i < aRegions.size(); ++i) {
		//aRegions.at(i).x = (int)((double)(aRegions.at(i).x-1) / (dScaling));
		aRegions[i].x = (int)((double)(aRegions[i].x-1) / (dScaling));
		aRegions[i].y = (int)((double)(aRegions[i].y-1) / (dScaling));
		aRegions[i].width = (int)((double)(aRegions[i].width+2) / (dScaling));
		aRegions[i].height = (int)((double)(aRegions[i].height+2) / (dScaling));
		if (aRegions[i].x < 0)		aRegions[i].x = 0;
		if (aRegions[i].y < 0)		aRegions[i].y = 0;
		if (aRegions[i].width + aRegions[i].x >= oImage.cols)	aRegions[i].width = oImage.cols - aRegions[i].x - 1;
		if (aRegions[i].height + aRegions[i].y > oImage.rows)	aRegions[i].height = oImage.rows - aRegions[i].y - 1;

		aMatRegions[i] = oImage(aRegions[i]);
		aDBScan[i] = oDBScan.Segment(aMatRegions[i]);
		//aDBScan[i] = oSegment.Segment(aMatRegions[i]);
		aDBScan[i] *= (double)(3 * (i+1));
		//oImageTwoStep(aRegions[i]) = aDBScan[i];
		aDBScan[i].copyTo(oImageTwoStep(aRegions[i]));

		normalize(aDBScan[i], aDBScan[i], 0.0, 255.0, CV_MINMAX);
		applyColorMap(aDBScan[i], aDBScan[i], COLORMAP_JET);

		
	}

	normalize(oRegionCoarse, oRegionCoarse, 0.0, 255.0, CV_MINMAX);
	normalize(oImageTwoStep, oImageTwoStep, 0.0, 255.0, CV_MINMAX);
	applyColorMap(oRegionCoarse, oRegionCoarse, COLORMAP_JET);
	applyColorMap(oImageTwoStep, oImageTwoStep, COLORMAP_JET);

	resize(oImageCoarse, oImageCoarse, oImage.size(), INTER_NEAREST);
	resize(oRegionCoarse, oRegionCoarse, oImage.size(), INTER_NEAREST);

	for (int i = 0; i < aRegions.size(); ++i) {
		double dColor = (double)(i)*50.0+50.0;
		rectangle(oRegionCoarse, aRegions[i], Scalar(0.0, 0.0, dColor));
		rectangle(oImage, aRegions[i], Scalar(255.0));

		string sText = "Region " + to_string(i);
		string sText2 = "DBSCAN " + to_string(i);
		imshow(sText2, aDBScan[i]);
		imshow(sText, aMatRegions[i]);
	}

	imshow("Image", oImage);
	imshow("Coarse", oImageCoarse);
	imshow("Region", oRegionCoarse);
	imshow("Two Step", oImageTwoStep);

	waitKey();

	return 0;
}

std::vector<cv::Rect2i> ExtractRegions(const cv::Mat& rCoarseRegions, int iNumRegions) {
	//cout << "Mat Type: " << type2str(rCoarseRegions.type()) << endl;
	//cout << rCoarseRegions << endl;
	vector<cv::Rect2i> aResult(iNumRegions);

	for (int i = 0; i < aResult.size(); ++i) {
		aResult[i] = Rect2i(-1, -1, -1, -1);
	}

	for (int i = 0; i < rCoarseRegions.rows; ++i) {
		for (int j = 0; j < rCoarseRegions.cols; ++j) {
			if (rCoarseRegions.at<uchar>(i, j) > 0) {
				int iRegionIndex = (int)(rCoarseRegions.at<uchar>(i, j))-1;
				
				Rect2i& rRegion = aResult[iRegionIndex];
				if (rRegion.x == -1) {
					rRegion.x = j;
					rRegion.width = j;
					rRegion.y = i;
					rRegion.height = i;
				}
				else {
					if (j < rRegion.x) {
						rRegion.x = j;
					}
					if (j > rRegion.width) {
						rRegion.width = j;
					}
					if (i < rRegion.y) {
						rRegion.y = i;
					}
					if (i > rRegion.height) {
						rRegion.height = i;
					}
				}
			}
		}
	}

	for (int i = 0; i < aResult.size(); ++i) {
		aResult[i].width = aResult[i].width - aResult[i].x+1;
		aResult[i].height = aResult[i].height - aResult[i].y+1;
	}

	return aResult;
}

std::vector<cv::Rect2i> MergeRegions(const std::vector<cv::Rect2i>& aRegions) {
	vector<cv::Rect2i> aRegionCopy(aRegions);
	vector<cv::Rect2i> aResult;

	for (size_t i = 0; i < aRegionCopy.size(); ++i) {
		for (size_t j = i + 1; j < aRegionCopy.size(); ++j) {
			if ((aRegionCopy[i] & aRegionCopy[j]).area() > 0) {
				aRegionCopy[i] = (aRegionCopy[i] | aRegionCopy[j]);
				aRegionCopy[j] = cv::Rect2i();
			}
		}
	}

	for (size_t i = 0; i < aRegionCopy.size(); ++i) {
		if (aRegionCopy[i].area() > 0) {
			aResult.push_back(aRegionCopy[i]);
		}
	}

	return aResult;
}
