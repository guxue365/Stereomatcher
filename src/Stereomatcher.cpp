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

#include "ImageControl.h"

#include "IImageLoader.h"
#include "IPreprocessing.h"
#include "IBackgroundSubtraction.h"
#include "ISegmentation.h"
#include "IPostprocessing.h"
#include "IStereoMatch.h"

#include <ConfigDef.h>

#include "imageloader/BaseImageloader.h"
#include "imageloader/CustomImageloader.h"

#include "preprocess/BasePreprocessor.h"
#include "preprocess/PreprocessMask.h"

#include "bgsubtraction/CustomPixelBasedAdaptiveSegmenter.h"
#include "bgsubtraction/CustomFrameDifference.h"

#include <postprocess/BasePostprocessor.h>
#include <postprocess/PostInterpolation.h>

#include "segmentation/RegionGrowing.h"
#include "segmentation/DBSCAN.h"

#include <stereomatch/BasicBlockMatcher.h>
#include <stereomatch/BasicBPMatcher.h>
#include <stereomatch/BasicSGMatcher.h>
#include <stereomatch/CustomBlockMatcher.h>
#include <stereomatch/CustomPyramidMatcher.h>
#include <stereomatch/CustomDiffMatcher.h>
#include <stereomatch/CustomCannyMatcher.h>

using namespace std;
using namespace cv;
using json = nlohmann::json;


int main() {

	vector<Run> aRuns;
	BaseImageloader oBaseImageloader;
	CustomImageloader oCustomImageloader;

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
	CustomPyramidMatcher oCustomPyramidMatcher(&oBasicBPMatcher);

	BasePostprocessor oBasePostprocessor;
	PostInterpolation oPostInterpolation;

	RegionGrowing oRegionGrowing;
	DBSCAN oDBSCAN;

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

		aRuns.push_back(oRun);
	}

	try {
		for(auto& rRun: aRuns) {
			cout<<endl<<"---------------------------------------------------------------"<<endl;
			cout<<"Starting run: "<<rRun.msTitle<<endl<<endl;
			cout<<"Image folder: "<<rRun.msImagefolder<<endl;

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
				default: throw std::invalid_argument("Invalid Segmentation");
			}

			ImageControl oImageControl(*pImageloader, *pPreprocessor, *pBackgroundSubtraction, *pStereomatch, *pPostprocessor, *pSegmentation);

			cout<<"Starting Computation"<<endl;
			auto oTimestart = chrono::high_resolution_clock::now();

			oImageControl.Run();

			auto oTimeend = chrono::high_resolution_clock::now();
			chrono::duration<double> oDuration = oTimeend-oTimestart;

			cout<<"Computation finished. Time taken: "<<oDuration.count()<<"s"<<endl;

			cout<<"Creating result folder: "<<rRun.msResultfolder<<endl;

			std::string sPreprocessFolder = rRun.msResultfolder+"preprocess/";
			std::string sForegroundFolder = rRun.msResultfolder+"foreground/";
			std::string sDisparityFolder = rRun.msResultfolder+"disparity/";
			std::string sPostprocessFolder = rRun.msResultfolder+"postprocess/";
			std::string sSegmentationFolder = rRun.msResultfolder+"segmentation/";

			if(bWriteResult) {
				mkdir(rRun.msResultfolder.c_str(), ACCESSPERMS);
				mkdir(sPreprocessFolder.c_str(), ACCESSPERMS);
				mkdir(sForegroundFolder.c_str(), ACCESSPERMS);
				mkdir(sDisparityFolder.c_str(), ACCESSPERMS);
				mkdir(sPostprocessFolder.c_str(), ACCESSPERMS);
				mkdir(sSegmentationFolder.c_str(), ACCESSPERMS);

				for(size_t i=0; i<oImageControl.getPreprocessLeft().size(); ++i) {
					std::string sFilename = "img_"+std::to_string(i)+".png";
					std::string sNameLeft = "img_"+std::to_string(i)+"_c0.png";
					std::string sNameRight = "img_"+std::to_string(i)+"_c1.png";

					imwrite(sPreprocessFolder+sNameLeft, oImageControl.getPreprocessLeft()[i]);
					imwrite(sPreprocessFolder+sNameRight, oImageControl.getPreprocessRight()[i]);

					imwrite(sForegroundFolder+sNameLeft, oImageControl.getForegroundLeft()[i]);
					imwrite(sForegroundFolder+sNameRight, oImageControl.getForegroundRight()[i]);

					imwrite(sDisparityFolder+sFilename, oImageControl.getDisparity()[i]);

					imwrite(sPostprocessFolder+sFilename, oImageControl.getPostprocessImages()[i]);

					imwrite(sSegmentationFolder+sFilename, oImageControl.getSegmentation()[i]);
				}
			}
		}

	} catch(std::exception& rEx) {
		cout<<"Exception: "<<rEx.what()<<endl;
	}

	cout<<"Finished"<<endl;

	return 0;
}
