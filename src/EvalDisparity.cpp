#include <iostream>
#include <vector>
//#include <Windows.h>
#include <sys/stat.h>

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
				if (!oJsonOptions["disparityrange"].empty()) {
					oRun.moStereoOptions.miDisparityRange = oJsonOptions["disparityrange"];
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
			CustomPyramidMatcher oCustomPyramidMatcher(&oBasicBPMatcher);

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
				oCustomBlockMatcher.setBlockSize(rRun.moStereoOptions.miBlocksize);
				oCustomDiffMatcher.setBlockSize(rRun.moStereoOptions.miBlocksize);
				oCustomCannyMatcher.setBlockSize(rRun.moStereoOptions.miBlocksize);
				oCustomPyramidMatcher.setBlockSize(rRun.moStereoOptions.miBlocksize);

				cout << "- Blocksize: " << rRun.moStereoOptions.miBlocksize << endl;
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

			//CreateDirectory(rRun.msResultfolder.c_str(), NULL);
			mkdir(rRun.msResultfolder.c_str(), ACCESSPERMS);
			
			vector<double> aBPPError;
			vector<double> aRMSError;

			for (int iFrame = 0; ; ++iFrame) {
				oFilestreamLeft >> oFrameLeftColor;
				oFilestreamRight >> oFrameRightColor;
				oFilestreamGT >> oDisparityGT;
				//oDisparityGT.convertTo(oDisparityGT, CV_8U, 1.0 / 256.0);
				cvtColor(oDisparityGT, oDisparityGT, CV_BGR2GRAY);
				oDisparityGT.convertTo(oDisparityGT, CV_8U, 1.0);

				if (oFrameLeftColor.empty() || oFrameRightColor.empty() || oDisparityGT.empty())	break;

				cvtColor(oFrameLeftColor, oFrameLeftGray, COLOR_BGR2GRAY);
				cvtColor(oFrameRightColor, oFrameRightGray, COLOR_BGR2GRAY);

				oCustomDisparity = pStereomatch->Match(oFrameLeftGray, oFrameRightGray);

				oCustomDisparity = pPostprocessor->Postprocess(oCustomDisparity);

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

		}

		
		
	}
	catch (std::exception& e) {
		cout << "Exception occurred: " << e.what() << endl;
		//system("pause");
		return -1;
	}

	//system("pause");

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
