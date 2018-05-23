#include <imageloader/BaseImageloader.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include <sys/stat.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <nlohmann/json.hpp>

#include "ImageControl.h"

#include "IImageLoader.h"
#include "IPreprocessing.h"
#include "IPostprocessing.h"
#include "IStereoMatch.h"

#include "imageloader/BaseImageloader.h"
#include "preprocess/BasePreprocessor.h"
#include "postprocess/BasePostprocessor.h"
#include "stereomatch/BasicBlockmatcher.h"
#include "stereomatch/BasicSGMatcher.h"
#include "stereomatch/BasicBPMatcher.h"

using namespace std;
using namespace cv;
using json = nlohmann::json;

enum E_IMAGELOADER {
	LOADER_BASE
};

enum E_PREPROCESSOR {
	PREPROC_BASE
};

enum E_POSTPROCESSOR {
	POSTPROC_BASE
};

enum E_STEREOMATCHER {
	BASIC_BLOCK,  // opencv blockmatcher
	BASIC_SG, // opencv semiglobal matching
	BASIC_BP  // opencv belief propagation
};

struct Run {
	std::string msTitle;
	std::string msImagefolder;
	std::string msResultfolder;
	E_IMAGELOADER meImageloader;
	E_PREPROCESSOR mePreprocessor;
	E_POSTPROCESSOR mePostProcessor;
	E_STEREOMATCHER meStereomatcher;
};

E_IMAGELOADER convertImageloader(const std::string& sImageloader);
E_PREPROCESSOR convertPreprocessor(const std::string& sPreprocessor);
E_POSTPROCESSOR convertPostprocessor(const std::string& sPostprocessor);
E_STEREOMATCHER convertStereomatcher(const std::string& sStereomatcher);

ostream& operator << (ostream& os, E_IMAGELOADER eImageloader);
ostream& operator << (ostream& os, E_PREPROCESSOR ePreprocess);
ostream& operator << (ostream& os, E_POSTPROCESSOR ePostProcess);
ostream& operator << (ostream& os, E_STEREOMATCHER eStereomatch);


int main() {

	vector<Run> aRuns;
	BaseImageloader oBaseImageloader;
	BasePreprocessor oBasePreprocessor;
	BasePostprocessor oBasePostprocessor;
	BasicBlockmatcher oBasicBlockmatcher;
	BasicSGMatcher oBasicSGMatcher;
	BasicBPMatcher oBasicBPMatcher;

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

	for(auto& oJsonRun: oJsonConfig["runs"]) {
		Run oRun;
		oRun.msTitle 			= oJsonRun["title"];
		oRun.msImagefolder 		= oJsonRun["image_folder"];
		oRun.msResultfolder 	= oJsonRun["result_folder"];

		oRun.meImageloader 		= convertImageloader(oJsonRun["image_loader"]);
		oRun.mePreprocessor 	= convertPreprocessor(oJsonRun["preprocessor"]);
		oRun.mePostProcessor 	= convertPostprocessor(oJsonRun["postprocessor"]);
		oRun.meStereomatcher 	= convertStereomatcher(oJsonRun["stereomatcher"]);

		aRuns.push_back(oRun);
	}

	try {
		for(auto& rRun: aRuns) {
			cout<<endl<<"---------------------------------------------------------------"<<endl;
			cout<<"Starting run: "<<rRun.msTitle<<endl<<endl;
			cout<<"Image folder: "<<rRun.msImagefolder<<endl;

			IImageLoader* pImageloader;
			IPreprocessing* pPreprocessor;
			IPostProcessing* pPostprocessor;
			IStereoMatch* pStereomatch;

			cout<<"Setting Imageloader: "<<rRun.meImageloader<<endl;
			switch(rRun.meImageloader) {
				case E_IMAGELOADER::LOADER_BASE: {
					pImageloader = &oBaseImageloader;
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
				default: {
					throw std::invalid_argument("Invalid Preprocessor");
				}
			}

			cout<<"Setting Postprocessor: "<<rRun.mePostProcessor<<endl;
			switch(rRun.mePostProcessor) {
				case E_POSTPROCESSOR::POSTPROC_BASE: {
					pPostprocessor = &oBasePostprocessor;
					break;
				}
				default: {
					throw std::invalid_argument("Invalid Postprocessor");
				}
			}

			cout<<"Setting Stereomatcher: "<<rRun.meStereomatcher<<endl;
			switch(rRun.meStereomatcher) {
				case E_STEREOMATCHER::BASIC_BLOCK: {
					pStereomatch = &oBasicBlockmatcher;
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
				default: {
					throw std::invalid_argument("Invalid Stereomatcher");
				}
			}

			ImageControl oImageControl(*pImageloader, *pPreprocessor, *pPostprocessor, *pStereomatch);

			cout<<"Starting stereo computation"<<endl;
			auto oTimestart = chrono::high_resolution_clock::now();

			oImageControl.Run();

			auto oTimeend = chrono::high_resolution_clock::now();
			chrono::duration<double> oDuration = oTimeend-oTimestart;

			cout<<"Computation finished. Time taken: "<<oDuration.count()<<"s"<<endl;

			cout<<"Creating result folder: "<<rRun.msResultfolder<<endl;
			std::string sPreprocessFolder = rRun.msResultfolder+"preprocess";
			std::string sForegroundFolder = rRun.msResultfolder+"foreground";
			std::string sDisparityFolder = rRun.msResultfolder+"disparity";
			std::string sPostprocessFolder = rRun.msResultfolder+"postprocess";

			mkdir(rRun.msResultfolder.c_str(), ACCESSPERMS);
			mkdir(sPreprocessFolder.c_str(), ACCESSPERMS);
			mkdir(sForegroundFolder.c_str(), ACCESSPERMS);
			mkdir(sDisparityFolder.c_str(), ACCESSPERMS);
			mkdir(sPostprocessFolder.c_str(), ACCESSPERMS);
		}

	} catch(std::exception& rEx) {
		cout<<"Exception: "<<rEx.what()<<endl;
	}

	cout<<"Finished"<<endl;

	return 0;
}

E_IMAGELOADER convertImageloader(const std::string& sImageloader) {
	if(sImageloader=="base") 	return E_IMAGELOADER::LOADER_BASE;
	throw std::invalid_argument("invalid imageloader conversion");
}

E_PREPROCESSOR convertPreprocessor(const std::string& sPreprocessor) {
	if(sPreprocessor=="base") 	return E_PREPROCESSOR::PREPROC_BASE;
	throw std::invalid_argument("invalid preprocessor conversion");
}

E_POSTPROCESSOR convertPostprocessor(const std::string& sPostprocessor) {
	if(sPostprocessor=="base") 	return E_POSTPROCESSOR::POSTPROC_BASE;
	throw std::invalid_argument("invalid postprocessor conversion");
}

E_STEREOMATCHER convertStereomatcher(const std::string& sStereomatcher) {
	if(sStereomatcher=="basicblock") 	return E_STEREOMATCHER::BASIC_BLOCK;
	if(sStereomatcher=="basicsg") 		return E_STEREOMATCHER::BASIC_SG;
	if(sStereomatcher=="basicbp") 		return E_STEREOMATCHER::BASIC_BP;
	throw std::invalid_argument("invalid stereomatcher conversion");
}

ostream& operator << (ostream& os, E_IMAGELOADER eImageloader) {
	switch(eImageloader) {
		case E_IMAGELOADER::LOADER_BASE: {
			os<<"base";
			break;
		}
	}
	return os;
}

ostream& operator << (ostream& os, E_PREPROCESSOR ePreprocess) {
	switch(ePreprocess) {
		case E_PREPROCESSOR::PREPROC_BASE: {
			os<<"base";
			break;
		}
	}
	return os;
}

ostream& operator << (ostream& os, E_POSTPROCESSOR ePostProcess) {
	switch(ePostProcess) {
		case E_POSTPROCESSOR::POSTPROC_BASE: {
			os<<"base";
			break;
		}
	}
	return os;
}

ostream& operator << (ostream& os, E_STEREOMATCHER eStereomatch) {
	switch(eStereomatch) {
		case E_STEREOMATCHER::BASIC_BLOCK: {
			os<<"basicblock";
			break;
		}
		case E_STEREOMATCHER::BASIC_SG: {
			os<<"basicsg";
			break;
		}
		case E_STEREOMATCHER::BASIC_BP: {
			os<<"basicbp";
			break;
		}
	}
	return os;
}
