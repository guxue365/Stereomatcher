#include <imageloader/BaseImageloader.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include <sys/stat.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <nlohmann/json.hpp>
#include <stereomatch/BasicBlockMatcher.h>

#include <ImageControl.h>

#include <IImageLoader.h>
#include <IPreprocessing.h>
#include <IBackgroundSubtraction.h>
#include <ISegmentation.h>
#include <IPostprocessing.h>
#include <IStereoMatch.h>

#include <ConfigDef.h>

#include <imageloader/BaseImageloader.h>
#include <imageloader/CustomImageloader.h>
#include <imageloader/SkipImageloader.h>

#include <preprocess/BasePreprocessor.h>
#include <preprocess/PreprocessMask.h>

#include <bgsubtraction/CustomPixelBasedAdaptiveSegmenter.h>
#include <bgsubtraction/CustomFrameDifference.h>

#include <postprocess/BasePostprocessor.h>
#include <postprocess/PostInterpolation.h>

#include <segmentation/RegionGrowing.h>
#include <segmentation/DBSCAN.h>
#include <segmentation/PCLSegmentation.h>
#include <segmentation/TwoStepSegmentation.h>

#include <stereomatch/BasicBlockMatcher.h>
#include <stereomatch/BasicBPMatcher.h>
#include <stereomatch/BasicSGMatcher.h>
#include <stereomatch/CustomBlockMatcher.h>
#include <stereomatch/CustomPyramidMatcher.h>
#include <stereomatch/CustomDiffMatcher.h>
#include <stereomatch/CustomCannyMatcher.h>
#include <stereomatch/CustomMultiBoxMatcher.h>
#include <stereomatch/CustomBlockCannyMatcher.h>

using namespace std;
using namespace cv;
using json = nlohmann::json;

void SaveClusterToJson(const std::string& sFilename, const std::vector<std::vector<Cluster> >& aFrameClusters);

int main() {

	vector<Run> aRuns;

	ifstream oConfigFile("config.json", ios::in);
	if(!oConfigFile.is_open()) {
		cout<<"Error: Cannot open config file"<<endl;
		return -1;
	}

	json oJsonConfig;

	oConfigFile>>oJsonConfig;

	oConfigFile.close();

	cout<<"Loading config.json"<<endl;
	cout<<"Version: "<<oJsonConfig["version"]<<endl;
	string sTitle = oJsonConfig["title"];
	cout<<"Title: "<<sTitle<<endl;
	bool bWriteResult = oJsonConfig["write_result"];
	cout<<"Write Result: "<<(bWriteResult ? "true" : "false")<<endl;
	bool bSkipBGS = oJsonConfig["skip_bgs"];
	cout<<"Skip BGS: "<<(bSkipBGS ? "true" : "false")<<endl;

	for(auto& oJsonRun: oJsonConfig["runs"]) {
		Run oRun;
		oRun.msTitle 			= oJsonRun["title"];
		oRun.msImagefolder 		= oJsonRun["image_folder"];
		oRun.msResultfolder 	= oJsonRun["result_folder"];

		oRun.meImageloader 		= convertImageloader(oJsonRun["image_loader"]);
		oRun.mePreprocessor 	= convertPreprocessor(oJsonRun["preprocessor"]);
		oRun.meBGSubtractor 	= convertBGSubtractor(oJsonRun["bgsubtraction"]);
		oRun.meStereomatcher 	= convertStereomatcher(oJsonRun["stereomatcher"]);
		oRun.mePostProcessor 	= convertPostprocessor(oJsonRun["postprocessor"]);
		oRun.meSegmentation 	= convertSegmentation(oJsonRun["segmentation"]);

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
			if(!oJsonOptions["scalingboxwidth"].empty()) {
				oRun.moStereoOptions.mdBoxScalingWidth = oJsonOptions["scalingboxwidth"];
			}
			if(!oJsonOptions["scalingboxheight"].empty()) {
				oRun.moStereoOptions.mdBoxScalingHeight = oJsonOptions["scalingboxheight"];
			}
		}

		aRuns.push_back(oRun);
	}

	try {
		for(auto& rRun: aRuns) {
			cout<<endl<<"---------------------------------------------------------------"<<endl;
			cout<<"Starting run: "<<rRun.msTitle<<endl<<endl;
			cout<<"Image folder: "<<rRun.msImagefolder<<endl;

			BaseImageloader oBaseImageloader;
			CustomImageloader oCustomImageloader;
			SkipImageloader oSkipImageloader;

			BasePreprocessor oBasePreprocessor;
			PreprocessMask oPreprocessMask;

			CustomPixelBasedAdaptiveSegmenter oPBAS;
			CustomFrameDifference oFD;

			BasicBlockMatcher oBasicBlockMatcher;
			BasicSGMatcher oBasicSGMatcher;
			BasicBPMatcher oBasicBPMatcher;
			CustomBlockMatcher oCustomBlockMatcher;
			CustomDiffMatcher oCustomDiffMatcher;
			CustomCannyMatcher oCustomCannyMatcher;
			CustomPyramidMatcher oCustomPyramidMatcher(&oCustomBlockMatcher);
			CustomMultiBoxMatcher oCustomMultiBoxMatcher;
			CustomBlockCannyMatcher oCustomBlockCannyMatcher;

			BasePostprocessor oBasePostprocessor;
			PostInterpolation oPostInterpolation;

			RegionGrowing oRegionGrowing;
			DBSCAN oDBSCAN;
			PCLSegmentation oPCLSegmentation;
			TwoStepSegmentation oTwoStepSegmentation;

			IImageLoader* pImageloader;
			IPreprocessing* pPreprocessor;
			IBackgroundSubtraction* pBackgroundSubtraction;
			IStereoMatch* pStereomatch;
			IPostProcessing* pPostprocessor;
			ISegmentation* pSegmentation;

			cout<<"Setting Imageloader: "<<rRun.meImageloader<<endl;
			switch(rRun.meImageloader) {
				case E_IMAGELOADER::LOADER_BASE: {
					pImageloader = &oBaseImageloader;
					break;
				}
				case E_IMAGELOADER::LOADER_CUSTOM: {
					pImageloader = &oCustomImageloader;
					break;
				}
				case E_IMAGELOADER::LOADER_SKIP: {
					pImageloader = &oSkipImageloader;
					break;
				}
				default: {
					throw std::invalid_argument("Invalid Imageloader");
				}
			}

			cout<<"Initialising Imageloader: ";
			bool bInitLoader = pImageloader->Init(rRun.msImagefolder);
			cout<<(bInitLoader ? "successful" : "failed")<<endl;
			if(!bInitLoader) {
				throw std::invalid_argument("Error initialising loader");
			}

			cout<<"Setting Preprocessor: "<<rRun.mePreprocessor<<endl;
			switch(rRun.mePreprocessor) {
				case E_PREPROCESSOR::PREPROC_BASE: {
					pPreprocessor = &oBasePreprocessor;
					break;
				}
				case E_PREPROCESSOR::PREPROC_MASK: {
					pPreprocessor = &oPreprocessMask;
					break;
				}
				default: {
					throw std::invalid_argument("Invalid Preprocessor");
				}
			}

			cout<<"Setting Background Subtraction: "<<rRun.meBGSubtractor<<endl;
			switch(rRun.meBGSubtractor) {
				case E_BGSUBTRACTOR::BG_PBAS: {
					pBackgroundSubtraction = &oPBAS;
					break;
				}
				case E_BGSUBTRACTOR::BG_FD: {
					pBackgroundSubtraction = &oFD;
					break;
				}
				default: {
					throw std::invalid_argument("Invalid Background Subtraction");
				}
			}

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

				oCustomMultiBoxMatcher.setBlockWidth(rRun.moStereoOptions.miBlocksize);
				oCustomMultiBoxMatcher.setBlockHeight(rRun.moStereoOptions.miBlocksize);

				oCustomBlockCannyMatcher.setBlockWidth(rRun.moStereoOptions.miBlocksize);
				oCustomBlockCannyMatcher.setBlockHeight(rRun.moStereoOptions.miBlocksize);

				cout << "- Blocksize: " << rRun.moStereoOptions.miBlocksize << endl;
			}
			if(rRun.moStereoOptions.miBlockWidth!=0) {
				oCustomBlockMatcher.setBlockWidth(rRun.moStereoOptions.miBlockWidth);
				oCustomDiffMatcher.setBlockWidth(rRun.moStereoOptions.miBlockWidth);
				oCustomCannyMatcher.setBlockWidth(rRun.moStereoOptions.miBlockWidth);
				oCustomPyramidMatcher.setBlockWidth(rRun.moStereoOptions.miBlockWidth);
				oCustomMultiBoxMatcher.setBlockWidth(rRun.moStereoOptions.miBlockWidth);
				oCustomBlockCannyMatcher.setBlockWidth(rRun.moStereoOptions.miBlockWidth);

				cout<<"- Block Width: "<<rRun.moStereoOptions.miBlockWidth;
			}
			if(rRun.moStereoOptions.miBlockHeight!=0) {
				oCustomBlockMatcher.setBlockHeight(rRun.moStereoOptions.miBlockHeight);
				oCustomDiffMatcher.setBlockHeight(rRun.moStereoOptions.miBlockHeight);
				oCustomCannyMatcher.setBlockHeight(rRun.moStereoOptions.miBlockHeight);
				oCustomPyramidMatcher.setBlockHeight(rRun.moStereoOptions.miBlockHeight);
				oCustomMultiBoxMatcher.setBlockHeight(rRun.moStereoOptions.miBlockHeight);
				oCustomBlockCannyMatcher.setBlockHeight(rRun.moStereoOptions.miBlockHeight);

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
				oCustomMultiBoxMatcher.setNumDisparities(rRun.moStereoOptions.miDisparityRange);
				oCustomBlockCannyMatcher.setNumDisparities(rRun.moStereoOptions.miDisparityRange);

				cout << "- Disparity Range: " << rRun.moStereoOptions.miDisparityRange << endl;
			}
			if(rRun.moStereoOptions.mdValidTolerance!=0.0) {
				oCustomBlockMatcher.setValidTolerance(rRun.moStereoOptions.mdValidTolerance);
				oCustomDiffMatcher.setValidTolerance(rRun.moStereoOptions.mdValidTolerance);
				oCustomCannyMatcher.setValidTolerance(rRun.moStereoOptions.mdValidTolerance);
				oCustomPyramidMatcher.setValidTolerance(rRun.moStereoOptions.mdValidTolerance);
				oCustomMultiBoxMatcher.setValidTolerance(rRun.moStereoOptions.mdValidTolerance);
				oCustomBlockCannyMatcher.setValidTolerance(rRun.moStereoOptions.mdValidTolerance);

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
				oCustomBlockCannyMatcher.setThreshold1(rRun.moStereoOptions.mdCannyThreshold1);

				cout << "- Canny Threshold1: " << rRun.moStereoOptions.mdCannyThreshold1 << endl;
			}
			if (rRun.moStereoOptions.mdCannyThreshold2 != 0.0) {
				oCustomCannyMatcher.setThreshold2(rRun.moStereoOptions.mdCannyThreshold2);
				oCustomBlockCannyMatcher.setThreshold2(rRun.moStereoOptions.mdCannyThreshold2);

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
			if(rRun.moStereoOptions.mdBoxScalingWidth != 0.0) {
				oCustomMultiBoxMatcher.setBoxWidthScaling(rRun.moStereoOptions.mdBoxScalingWidth);

				cout<<"- Box Width Scaling: "<<rRun.moStereoOptions.mdBoxScalingWidth<<endl;
			}
			if(rRun.moStereoOptions.mdBoxScalingHeight != 0.0) {
				oCustomMultiBoxMatcher.setBoxHeightScaling(rRun.moStereoOptions.mdBoxScalingHeight);

				cout<<"- Box Height Scaling: "<<rRun.moStereoOptions.mdBoxScalingHeight<<endl;
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

			cout<<"Setting Segmentation: "<<rRun.meSegmentation<<endl;
			switch(rRun.meSegmentation) {
				case E_SEGMENTATION::E_REGIONGROWING: {
					pSegmentation = &oRegionGrowing;
					break;
				}
				case E_SEGMENTATION::E_DBSCAN: {
					pSegmentation = &oDBSCAN;
					break;
				}
				case E_SEGMENTATION::E_PCLSEGMENTATION: {
					pSegmentation = &oPCLSegmentation;
					break;
				}
				case E_SEGMENTATION::E_TWOSTEPSEGMENTATION: {
					pSegmentation = &oTwoStepSegmentation;
					break;
				}
				default: throw std::invalid_argument("Invalid Segmentation");
			}



			ImageControl oImageControl(*pImageloader, *pPreprocessor, *pBackgroundSubtraction, *pStereomatch, *pPostprocessor, *pSegmentation);

			cout<<"Starting Computation"<<endl;
			auto oTimestart = chrono::high_resolution_clock::now();

			oImageControl.Run(bSkipBGS);

			auto oTimeend = chrono::high_resolution_clock::now();
			chrono::duration<double> oDuration = oTimeend-oTimestart;

			cout<<"Computation finished. Time taken: "<<oDuration.count()<<"s"<<endl;

			cout<<"Creating result folder: "<<rRun.msResultfolder<<endl;

			std::string sOriginalFolder = rRun.msResultfolder+"original/";
			std::string sPreprocessFolder = rRun.msResultfolder+"preprocess/";
			std::string sForegroundFolder = rRun.msResultfolder+"foreground/";
			std::string sDisparityFolder = rRun.msResultfolder+"disparity/";
			std::string sPostprocessFolder = rRun.msResultfolder+"postprocess/";

			if(bWriteResult) {
				mkdir(rRun.msResultfolder.c_str(), ACCESSPERMS);
				mkdir(sOriginalFolder.c_str(), ACCESSPERMS);
				mkdir(sPreprocessFolder.c_str(), ACCESSPERMS);
				mkdir(sForegroundFolder.c_str(), ACCESSPERMS);
				mkdir(sDisparityFolder.c_str(), ACCESSPERMS);
				mkdir(sPostprocessFolder.c_str(), ACCESSPERMS);

				for(size_t i=0; i<oImageControl.getPreprocessLeft().size(); ++i) {
					std::string sFilename = "img_"+std::to_string(i)+".png";
					std::string sNameLeft = "img_"+std::to_string(i)+"_c0.png";
					std::string sNameRight = "img_"+std::to_string(i)+"_c1.png";

					imwrite(sOriginalFolder+sNameLeft, oImageControl.getLeftImages()[i]);
					imwrite(sOriginalFolder+sNameRight, oImageControl.getRightImages()[i]);

					imwrite(sPreprocessFolder+sNameLeft, oImageControl.getPreprocessLeft()[i]);
					imwrite(sPreprocessFolder+sNameRight, oImageControl.getPreprocessRight()[i]);

					imwrite(sForegroundFolder+sNameLeft, oImageControl.getForegroundLeft()[i]);
					imwrite(sForegroundFolder+sNameRight, oImageControl.getForegroundRight()[i]);

					imwrite(sDisparityFolder+sFilename, oImageControl.getDisparity()[i]);

					imwrite(sPostprocessFolder+sFilename, oImageControl.getPostprocessImages()[i]);
				}

				SaveClusterToJson(rRun.msResultfolder+"result_cluster.json", oImageControl.getCluster());
			}
		}

	} catch(std::exception& rEx) {
		cout<<"Exception: "<<rEx.what()<<endl;
	}

	cout<<"Finished"<<endl;

	return 0;
}

void SaveClusterToJson(const std::string& sFilename, const std::vector<std::vector<Cluster> >& aFrameClusters) {
	json aJsonFrames;

	int iCount = 0;
	for(int i=0; i<aFrameClusters.size(); ++i) {
		auto& aCluster = aFrameClusters[i];

		json aJsonClusters;
		for(auto& oCluster: aCluster) {
			json oJsonCluster;
			oJsonCluster["id"] = iCount;
			oJsonCluster["frame"] = i;
			oJsonCluster["position"] = {oCluster.oPosition.val[0], oCluster.oPosition.val[1], oCluster.oPosition.val[2]};
			oJsonCluster["dimension"] = {oCluster.aDimension.val[0], oCluster.aDimension.val[1], oCluster.aDimension.val[2]};
			oJsonCluster["eccentricity"] = {oCluster.aEccentricity.val[0], oCluster.aEccentricity.val[1]};

			aJsonClusters.push_back(oJsonCluster);
			++iCount;
		}

		aJsonFrames.push_back(aJsonClusters);
	}

	ofstream oJsonFile(sFilename.c_str(), ios::out);
	if (!oJsonFile.is_open()) {
		throw std::invalid_argument("Cannot save GT File");
	}

	oJsonFile << aJsonFrames.dump(5);

	oJsonFile.close();
}
