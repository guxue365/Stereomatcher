#include <iostream>
#include <vector>
#include <Windows.h>
#include <sys/stat.h>
#include <chrono>
#include <ctime>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <nlohmann/json.hpp>

#include <ConfigDef.h>

#include <stereomatch/BasicBlockMatcher.h>
#include <stereomatch/BasicBPMatcher.h>
#include <stereomatch/BasicSGMatcher.h>
#include <stereomatch/CustomBlockMatcher.h>
#include <stereomatch/CustomPyramidMatcher.h>
#include <stereomatch/CustomDiffMatcher.h>
#include <stereomatch/CustomCannyMatcher.h>

#include <postprocess/BasePostprocessor.h>
#include <postprocess/PostInterpolation.h>

#include <evaluation/EvaluateBPP.h>
#include <evaluation/EvaluateRMS.h>

using namespace std;
using namespace cv;
using json = nlohmann::json;

std::tuple<double, double> ComputeMeanVar(const std::vector<double>& aValues);

int main() {
	vector<RunEvalDisparity> aRuns;

	ifstream oConfigFile("config_eval_disp.json", ios::in);
	if (!oConfigFile.is_open()) {
		cout << "Error: Cannot open config file" << endl;
		return -1;
	}

	json oJsonConfig;

	oConfigFile >> oJsonConfig;

	oConfigFile.close();

	try {
		cout << "Loading config.json" << endl;
		cout << "Version: " << oJsonConfig["version"] << endl;
		string sTitle = oJsonConfig["title"];
		cout << "Title: " << sTitle << endl;

		for (json oJsonRun : oJsonConfig["runs"]) {
			RunEvalDisparity oRun;
			std::string sTitle = oJsonRun["title"];
			std::string sImagefolder = oJsonRun["image_folder"];
			std::string sResultfolder = oJsonRun["result_folder"];

			oRun.msTitle = sTitle;
			oRun.msImagefolder = sImagefolder;
			oRun.msResultfolder = sResultfolder;

			oRun.meStereomatcher = convertStereomatcher(oJsonRun["stereomatcher"]);
			oRun.mePostProcessor = convertPostprocessor(oJsonRun["postprocessor"]);

			if (!oJsonRun["stereooptions"].empty()) {
				json& oJsonOptions = oJsonRun["stereooptions"];
				if (!oJsonOptions["blocksize"].empty()) {
					oRun.moStereoOptions.miBlocksize = oJsonOptions["blocksize"];
				}
				if (!oJsonOptions["blockwidth"].empty()) {
					oRun.moStereoOptions.miBlockWidth = oJsonOptions["blockwidth"];
				}
				if (!oJsonOptions["blockheight"].empty()) {
					oRun.moStereoOptions.miBlockHeight = oJsonOptions["blockheight"];
				}
				if (!oJsonOptions["disparityrange"].empty()) {
					oRun.moStereoOptions.miDisparityRange = oJsonOptions["disparityrange"];
				}
				if (!oJsonOptions["validtolerance"].empty()) {
					oRun.moStereoOptions.mdValidTolerance = oJsonOptions["validtolerance"];
				}
				if (!oJsonOptions["difforderx"].empty()) {
					oRun.moStereoOptions.miDiffOrderX = oJsonOptions["difforderx"];
				}
				if (!oJsonOptions["diffordery"].empty()) {
					oRun.moStereoOptions.miDiffOrderY = oJsonOptions["diffordery"];
				}
				if (!oJsonOptions["sobelkernelsize"].empty()) {
					oRun.moStereoOptions.miSobelKernelSize = oJsonOptions["sobelkernelsize"];
				}
				if (!oJsonOptions["cannythreshold1"].empty()) {
					oRun.moStereoOptions.mdCannyThreshold1 = oJsonOptions["cannythreshold1"];
				}
				if (!oJsonOptions["cannythreshold2"].empty()) {
					oRun.moStereoOptions.mdCannyThreshold2 = oJsonOptions["cannythreshold2"];
				}
				if (!oJsonOptions["scalingwidth"].empty()) {
					oRun.moStereoOptions.mdScalingWidth = oJsonOptions["scalingwidth"];
				}
				if (!oJsonOptions["scalingheight"].empty()) {
					oRun.moStereoOptions.mdScalingHeight = oJsonOptions["scalingheight"];
				}
			}

			aRuns.push_back(oRun);
		}

	
		for (auto& rRun : aRuns) {
			cout << endl << "---------------------------------------------------------------" << endl;
			cout << "Starting run: " << rRun.msTitle << endl << endl;
			cout << "Image folder: " << rRun.msImagefolder << endl;

			BasicBlockMatcher oBasicBlockMatcher;
			BasicSGMatcher oBasicSGMatcher;
			BasicBPMatcher oBasicBPMatcher;
			CustomBlockMatcher oCustomBlockMatcher;
			CustomDiffMatcher oCustomDiffMatcher;
			CustomCannyMatcher oCustomCannyMatcher;
			CustomPyramidMatcher oCustomPyramidMatcher(&oCustomBlockMatcher);

			BasePostprocessor oBasePostprocessor;
			PostInterpolation oPostInterpolation;

			EvaluateBPP oEvalBPP(10.0);
			EvaluateRMS oEvalRMS;

			IStereoMatch* pStereomatch;
			IPostProcessing* pPostprocessor;

			cout << "Setting Stereomatcher: " << rRun.meStereomatcher << endl;
			switch (rRun.meStereomatcher) {
				case E_STEREOMATCHER::BASIC_BLOCK: {
					pStereomatch = &oBasicBlockMatcher;
					break;
				}
				case E_STEREOMATCHER::BASIC_SG: {
					pStereomatch = &oBasicSGMatcher;
					break;
				}
				case E_STEREOMATCHER::BASIC_BP: {
					pStereomatch = &oBasicBPMatcher;
					break;
				}
				case E_STEREOMATCHER::CUSTOM_BLOCK: {
					pStereomatch = &oCustomBlockMatcher;
					break;
				}
				case E_STEREOMATCHER::CUSTOM_DIFF: {
					pStereomatch = &oCustomDiffMatcher;
					break;
				}
				case E_STEREOMATCHER::CUSTOM_CANNY: {
					pStereomatch = &oCustomCannyMatcher;
					break;
				}
				case E_STEREOMATCHER::CUSTOM_PYRAMID: {
					pStereomatch = &oCustomPyramidMatcher;
					break;
				}
				default: {
					throw std::invalid_argument("Invalid Stereomatcher");
				}
			}

			cout << "Setting Stereo Options: " << endl;
			if (rRun.moStereoOptions.miBlocksize != 0) {
				oBasicBlockMatcher.setBlockSize(rRun.moStereoOptions.miBlocksize);
				oBasicSGMatcher.setBlockSize(rRun.moStereoOptions.miBlocksize);

				oCustomBlockMatcher.setBlockWidth(rRun.moStereoOptions.miBlocksize);
				oCustomBlockMatcher.setBlockHeight(rRun.moStereoOptions.miBlocksize);

				oCustomDiffMatcher.setBlockWidth(rRun.moStereoOptions.miBlocksize);
				oCustomDiffMatcher.setBlockHeight(rRun.moStereoOptions.miBlocksize);

				oCustomCannyMatcher.setBlockWidth(rRun.moStereoOptions.miBlocksize);
				oCustomCannyMatcher.setBlockHeight(rRun.moStereoOptions.miBlocksize);

				oCustomPyramidMatcher.setBlockWidth(rRun.moStereoOptions.miBlocksize);
				oCustomPyramidMatcher.setBlockHeight(rRun.moStereoOptions.miBlocksize);

				cout << "- Blocksize: " << rRun.moStereoOptions.miBlocksize << endl;
			}
			if(rRun.moStereoOptions.miBlockWidth!=0) {
				oCustomBlockMatcher.setBlockWidth(rRun.moStereoOptions.miBlockWidth);
				oCustomDiffMatcher.setBlockWidth(rRun.moStereoOptions.miBlockWidth);
				oCustomCannyMatcher.setBlockWidth(rRun.moStereoOptions.miBlockWidth);
				oCustomPyramidMatcher.setBlockWidth(rRun.moStereoOptions.miBlockWidth);

				cout<<"- Block Width: "<<rRun.moStereoOptions.miBlockWidth;
			}
			if(rRun.moStereoOptions.miBlockHeight!=0) {
				oCustomBlockMatcher.setBlockHeight(rRun.moStereoOptions.miBlockHeight);
				oCustomDiffMatcher.setBlockHeight(rRun.moStereoOptions.miBlockHeight);
				oCustomCannyMatcher.setBlockHeight(rRun.moStereoOptions.miBlockHeight);
				oCustomPyramidMatcher.setBlockHeight(rRun.moStereoOptions.miBlockHeight);

				cout<<"- Block Height: "<<rRun.moStereoOptions.miBlockHeight;
			}
			if (rRun.moStereoOptions.miDisparityRange != 0) {
				oBasicBlockMatcher.setNumDisparities(rRun.moStereoOptions.miDisparityRange);
				oBasicSGMatcher.setNumDisparities(rRun.moStereoOptions.miDisparityRange);
				oBasicBPMatcher.setNumDisparities(rRun.moStereoOptions.miDisparityRange);
				oCustomBlockMatcher.setNumDisparities(rRun.moStereoOptions.miDisparityRange);
				oCustomDiffMatcher.setNumDisparities(rRun.moStereoOptions.miDisparityRange);
				oCustomCannyMatcher.setNumDisparities(rRun.moStereoOptions.miDisparityRange);
				oCustomPyramidMatcher.setNumDisparities(rRun.moStereoOptions.miDisparityRange);

				cout << "- Disparity Range: " << rRun.moStereoOptions.miDisparityRange << endl;
			}
			if(rRun.moStereoOptions.mdValidTolerance!=0.0) {
				oCustomBlockMatcher.setValidTolerance(rRun.moStereoOptions.mdValidTolerance);
				oCustomDiffMatcher.setValidTolerance(rRun.moStereoOptions.mdValidTolerance);
				oCustomCannyMatcher.setValidTolerance(rRun.moStereoOptions.mdValidTolerance);
				oCustomPyramidMatcher.setValidTolerance(rRun.moStereoOptions.mdValidTolerance);

				cout<<"- Valid Tolerance: "<<rRun.moStereoOptions.mdValidTolerance<<endl;
			}
			if(rRun.moStereoOptions.miDiffOrderX!=0) {
				oCustomDiffMatcher.setDiffOrderX(rRun.moStereoOptions.miDiffOrderX);

				cout<<"- Diff Order X: "<<rRun.moStereoOptions.miDiffOrderX;
			}
			if(rRun.moStereoOptions.miDiffOrderY!=0) {
				oCustomDiffMatcher.setDiffOrderY(rRun.moStereoOptions.miDiffOrderY);

				cout<<"- Diff Order Y: "<<rRun.moStereoOptions.miDiffOrderY;
			}
			if(rRun.moStereoOptions.miSobelKernelSize!=0) {
				oCustomDiffMatcher.setSobelKernelSize(rRun.moStereoOptions.miSobelKernelSize);

				cout<<"- Sobel Kernel Size: "<<rRun.moStereoOptions.miSobelKernelSize;
			}
			if (rRun.moStereoOptions.mdCannyThreshold1 != 0.0) {
				oCustomCannyMatcher.setThreshold1(rRun.moStereoOptions.mdCannyThreshold1);

				cout << "- Canny Threshold1: " << rRun.moStereoOptions.mdCannyThreshold1 << endl;
			}
			if (rRun.moStereoOptions.mdCannyThreshold2 != 0.0) {
				oCustomCannyMatcher.setThreshold2(rRun.moStereoOptions.mdCannyThreshold2);

				cout << "- Canny Threshold2: " << rRun.moStereoOptions.mdCannyThreshold2 << endl;
			}
			if (rRun.moStereoOptions.mdScalingWidth != 0.0) {
				oCustomPyramidMatcher.setScalingWidth(rRun.moStereoOptions.mdScalingWidth);

				cout << "- Scaling Width: " << rRun.moStereoOptions.mdScalingWidth << endl;
			}
			if (rRun.moStereoOptions.mdScalingHeight != 0.0) {
				oCustomPyramidMatcher.setScalingHeight(rRun.moStereoOptions.mdScalingHeight);

				cout << "- Scaling Height: " << rRun.moStereoOptions.mdScalingHeight << endl;
			}

			cout << "Setting Postprocessor: " << rRun.mePostProcessor << endl;
			switch (rRun.mePostProcessor) {
				case E_POSTPROCESSOR::POSTPROC_BASE: {
					pPostprocessor = &oBasePostprocessor;
					break;
				}
				case E_POSTPROCESSOR::POSTPROC_INTERPOLATION: {
					pPostprocessor = &oPostInterpolation;
					break;
				}
				default: {
					throw std::invalid_argument("Invalid Postprocessor");
				}
			}

			VideoCapture oFilestreamLeft(rRun.msImagefolder + "/image_2/%06d_10.png");
			VideoCapture oFilestreamRight(rRun.msImagefolder + "/image_3/%06d_10.png");
			VideoCapture oFilestreamGT(rRun.msImagefolder + "/disp_custom/%06d_10.png");

			if (!oFilestreamLeft.isOpened() || !oFilestreamRight.isOpened() || !oFilestreamGT.isOpened()) {
				throw std::invalid_argument("Error opening filestreams");
			}

			Mat oFrameLeftColor;
			Mat oFrameLeftGray;
			Mat oFrameRightColor;
			Mat oFrameRightGray;

			Mat oCustomDisparity;
			Mat oDisparityGT;

			Mat oEvalDispBPP;
			Mat oEvalDispRMS;

			CreateDirectory(rRun.msResultfolder.c_str(), NULL);
			//mkdir(rRun.msResultfolder.c_str(), ACCESSPERMS);
			
			vector<double> aBPPError;
			vector<double> aRMSError;
			long long iDuration = 0;

			for (int iFrame = 0; ; ++iFrame) {
				oFilestreamLeft >> oFrameLeftColor;
				oFilestreamRight >> oFrameRightColor;
				oFilestreamGT >> oDisparityGT;
				oDisparityGT.convertTo(oDisparityGT, CV_8U, 1.0 / 256.0);
				//cvtColor(oDisparityGT, oDisparityGT, CV_BGR2GRAY);
				//oDisparityGT.convertTo(oDisparityGT, CV_8U, 1.0);

				if (oFrameLeftColor.empty() || oFrameRightColor.empty() || oDisparityGT.empty())	break;

				cvtColor(oFrameLeftColor, oFrameLeftGray, COLOR_BGR2GRAY);
				cvtColor(oFrameRightColor, oFrameRightGray, COLOR_BGR2GRAY);

				auto tStart = std::chrono::high_resolution_clock::now();

				//oCustomDisparity = pStereomatch->Match(oFrameLeftGray, oFrameRightGray);
				oCustomDisparity = pStereomatch->Match(oFrameLeftColor, oFrameRightColor);

				oCustomDisparity = pPostprocessor->Postprocess(oCustomDisparity);

				auto tEnd = std::chrono::high_resolution_clock::now();
				long long iElapsedMicro = std::chrono::duration_cast<std::chrono::microseconds>(tEnd - tStart).count();
				iDuration += iElapsedMicro;

				double dEvalBPP = oEvalBPP.Evaluate(oDisparityGT, oCustomDisparity);
				oEvalDispBPP = oEvalBPP.getVisualRepresentation();

				double dEvalRMS = oEvalRMS.Evaluate(oDisparityGT, oCustomDisparity);
				oEvalDispRMS = oEvalRMS.getVisualRepresentation();

				aBPPError.push_back(dEvalBPP);
				aRMSError.push_back(dEvalRMS);

				std::string sFilenameBPP = rRun.msResultfolder + "/bpp_"+std::to_string(iFrame)+".png";
				std::string sFilenameRMS = rRun.msResultfolder + "/rms_" + std::to_string(iFrame) + ".png";
				std::string sFilenameOV = rRun.msResultfolder + "/overview_" + std::to_string(iFrame) + ".png";

				Mat oRow1;
				Mat oRow2;
				Mat oOverview;

				oCustomDisparity *= 3;
				oDisparityGT *= 3;
				oEvalDispRMS *= 3;

				oEvalDispRMS.convertTo(oEvalDispRMS, CV_8U);

				applyColorMap(oCustomDisparity, oCustomDisparity, COLORMAP_JET);
				applyColorMap(oDisparityGT, oDisparityGT, COLORMAP_JET);
				applyColorMap(oEvalDispRMS, oEvalDispRMS, COLORMAP_JET);

				hconcat(oDisparityGT, oFrameLeftColor, oRow1);
				hconcat(oCustomDisparity, oEvalDispRMS, oRow2);
				vconcat(oRow1, oRow2, oOverview);

				/*imshow("Overview", oOverview);
				cout << "BPP: " << dEvalBPP << endl;
				waitKey();*/

				imwrite(sFilenameBPP, oEvalDispBPP);
				imwrite(sFilenameRMS, oEvalDispRMS);
				imwrite(sFilenameOV, oOverview);

			}

			double dMeanBPP, dVarBPP;
			tie(dMeanBPP, dVarBPP) = ComputeMeanVar(aBPPError);

			double dMeanRMS, dVarRMS;
			tie(dMeanRMS, dVarRMS) = ComputeMeanVar(aRMSError);

			cout << "BPP: " << endl << " - Mean: " << dMeanBPP << endl << " - Var: " << dVarBPP << endl;
			cout << "RMS: " << endl << " - Mean: " << dMeanRMS << endl << " - Var: " << dVarRMS << endl;
			cout << "Duration: " << (double)(iDuration) / 1e6 <<"s"<< endl;
		}

		
		
	}
	catch (std::exception& e) {
		cout << "Exception occurred: " << e.what() << endl;
		system("pause");
		return -1;
	}

	system("pause");

	return 0;
}

std::tuple<double, double> ComputeMeanVar(const std::vector<double>& aValues) {
	double dMean = 0.0;

	for (auto& dValue : aValues) {
		dMean += dValue;
	}
	dMean /= (double)(aValues.size());

	double dVar = 0.0;

	for (auto& dValue : aValues) {
		dVar += (dMean - dValue)*(dMean - dValue);
	}
	dVar /= (double)(aValues.size());

	return make_tuple(dMean, dVar);
}
