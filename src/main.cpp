//============================================================================
// Name        : Vorlage.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include <sys/stat.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <boost/filesystem.hpp>

#include <nlohmann/json.hpp>

#include "ImageControl.h"

#include "ImageHandler.h"
#include "IPreprocessing.h"
#include "IPostprocessing.h"
#include "IStereoMatch.h"
#include "IStereoEvaluation.h"

#include "preprocess/BasePreprocessor.h"
#include "postprocess/BasePostprocessor.h"
#include "stereomatch/BasicBlockmatcher.h"
#include "stereomatch/BasicSGMatcher.h"
#include "evaluation/EvaluateRMS.h"
#include "evaluation/EvaluateBPP.h"

using namespace std;
using namespace cv;
using json = nlohmann::json;

enum E_PREPROCESSOR {
	PREPROC_BASE
};

enum E_POSTPROCESSOR {
	POSTPROC_BASE
};

enum E_STEREOMATCHER {
	BASIC_BLOCK,
	BASIC_SG
};

enum E_EVALUATION {
	EVAL_RMS,
	EVAL_BPP
};

struct Run {
	std::string msTitle;
	std::string msImagesLeft;
	std::string msImagesRight;
	std::string msImagesEvaluation;
	std::string msResultfolder;
	E_PREPROCESSOR mePreprocessor;
	E_POSTPROCESSOR mePostProcessor;
	E_STEREOMATCHER meStereomatcher;
	E_EVALUATION meEvaluation;
};

E_PREPROCESSOR convertPreprocessor(const std::string& sPreprocessor);
E_POSTPROCESSOR convertPostprocessor(const std::string& sPostprocessor);
E_STEREOMATCHER convertStereomatcher(const std::string& sStereomatcher);
E_EVALUATION convertEvaluation(const std::string& sEvaluation);

ostream& operator << (ostream& os, E_PREPROCESSOR ePreprocess);
ostream& operator << (ostream& os, E_POSTPROCESSOR ePostProcess);
ostream& operator << (ostream& os, E_STEREOMATCHER eStereomatch);
ostream& operator << (ostream& os, E_EVALUATION eEvaluation);


int main() {

	vector<Run> aRuns;
	BasePreprocessor oBasePreprocessor;
	BasePostprocessor oBasePostprocessor;
	BasicBlockmatcher oBasicBlockmatcher;
	BasicSGMatcher oBasicSGMatcher;
	EvaluateRMS oEvaluateRMS;
	EvaluateBPP oEvaluateBPP(2.0);


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
		oRun.msImagesLeft 		= oJsonRun["images_left"];
		oRun.msImagesRight 		= oJsonRun["images_right"];
		oRun.msImagesEvaluation = oJsonRun["images_evaluation"];
		oRun.msResultfolder 	= oJsonRun["result_folder"];

		oRun.mePreprocessor 	= convertPreprocessor(oJsonRun["preprocessor"]);
		oRun.mePostProcessor 	= convertPostprocessor(oJsonRun["postprocessor"]);
		oRun.meStereomatcher 	= convertStereomatcher(oJsonRun["stereomatcher"]);
		oRun.meEvaluation 		= convertEvaluation(oJsonRun["evaluation"]);

		aRuns.push_back(oRun);
	}

	try {
		for(auto& rRun: aRuns) {
			cout<<endl<<"---------------------------------------------------------------"<<endl;
			cout<<"Starting run: "<<rRun.msTitle<<endl<<endl;
			cout<<"Right image folder: "<<rRun.msImagesLeft<<endl;
			cout<<"Left image folder: "<<rRun.msImagesRight<<endl;
			cout<<"Stereo evaluation folder: "<<rRun.msImagesEvaluation<<endl;
			ImageHandler oImageHandler(rRun.msImagesLeft, rRun.msImagesRight, rRun.msImagesEvaluation, rRun.msResultfolder);

			IPreprocessing* pPreprocessor;
			IPostProcessing* pPostprocessor;
			IStereoMatch* pStereomatch;
			IStereoEvaluation* pStereoEvaluation;

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
				default: {
					throw std::invalid_argument("Invalid Stereomatcher");
				}
			}

			cout<<"Setting Evaluation: "<<rRun.meEvaluation<<endl;
			switch(rRun.meEvaluation) {
				case E_EVALUATION::EVAL_RMS: {
					pStereoEvaluation = &oEvaluateRMS;
					break;
				}
				case E_EVALUATION::EVAL_BPP: {
					pStereoEvaluation = &oEvaluateBPP;
					break;
				}
				default: {
					throw std::invalid_argument("Invalid Stereo Evaluation");
				}
			}

			ImageControl oImageControl(oImageHandler, *pPreprocessor, *pPostprocessor, *pStereomatch, *pStereoEvaluation);

			cout<<"Loading images"<<endl;
			oImageControl.LoadImages();

			cout<<"Starting stereo computation"<<endl;
			auto oTimestart = chrono::high_resolution_clock::now();

			oImageControl.Run();

			auto oTimeend = chrono::high_resolution_clock::now();
			chrono::duration<double> oDuration = oTimeend-oTimestart;

			cout<<"Computation finished. Time taken: "<<oDuration.count()<<"s"<<endl;

			cout<<"Creating result folder: "<<rRun.msResultfolder<<endl;
			mkdir(rRun.msResultfolder.c_str(), ACCESSPERMS);
			std::string sResultEvalFolder = rRun.msResultfolder+"eval/";
			cout<<"Creating result evaluation folder: "<<sResultEvalFolder<<endl;
			mkdir(sResultEvalFolder.c_str(), ACCESSPERMS);

			cout<<"Storing Results"<<endl;
			oImageControl.StoreResults();

			cout<<"Evaluating Results"<<endl;
			oImageControl.Evaluate();
		}

	} catch(std::string& sEx) {
		cout<<"Exception: "<<sEx<<endl;
	}

	cout<<"Finished"<<endl;

	return 0;
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
	throw std::invalid_argument("invalid stereomatcher conversion");
}

E_EVALUATION convertEvaluation(const std::string& sEvaluation) {
	if(sEvaluation=="rms") 		return E_EVALUATION::EVAL_RMS;
	if(sEvaluation=="bpp") 		return E_EVALUATION::EVAL_BPP;
	throw std::invalid_argument("invalid evaluation conversion");
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
	}
	return os;
}

ostream& operator << (ostream& os, E_EVALUATION eEvaluation) {
	switch(eEvaluation) {
		case E_EVALUATION::EVAL_RMS: {
			os<<"Root-Mean-Squared";
			break;
		}
		case E_EVALUATION::EVAL_BPP: {
			os<<"Bad-Pixel-Percentage";
			break;
		}
	}
	return os;
}
