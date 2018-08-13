#pragma once

#include <string>
#include <iostream>
#include <fstream>

enum E_IMAGELOADER {
	LOADER_BASE,
	LOADER_CUSTOM
};

enum E_PREPROCESSOR {
	PREPROC_BASE
};

enum E_BGSUBTRACTOR {
	BG_PBAS,
	BG_FD
};

enum E_STEREOMATCHER {
	BASIC_BLOCK,  // opencv blockmatcher
	BASIC_SG, // opencv semiglobal matching
	BASIC_BP,  // opencv belief propagation
	CUSTOM_BLOCK,  // custom block matcher
	CUSTOM_DIFF,  // custom diff matcher
	CUSTOM_CANNY, // custom canny matcher
	CUSTOM_PYRAMID  // custom pyramid matcher
};

enum E_POSTPROCESSOR {
	POSTPROC_BASE,
	POSTPROC_INTERPOLATION
};

enum E_SEGMENTATION {
	E_REGIONGROWING,
	E_DBSCAN
};

struct StereoOptions {
	int miBlocksize = 0;
	int miDisparityRange = 0;
};

struct Run {
	std::string msTitle;
	std::string msImagefolder;
	std::string msResultfolder;
	E_IMAGELOADER meImageloader;
	E_PREPROCESSOR mePreprocessor;
	E_BGSUBTRACTOR meBGSubtractor;
	E_STEREOMATCHER meStereomatcher;
	E_POSTPROCESSOR mePostProcessor;
	E_SEGMENTATION meSegmentation;
};

struct RunEvalDisparity {
	std::string msTitle;
	std::string msImagefolder;
	std::string msResultfolder;
	E_STEREOMATCHER meStereomatcher;
	StereoOptions moStereoOptions;
	E_POSTPROCESSOR mePostProcessor;
};

E_IMAGELOADER convertImageloader(const std::string& sImageloader) {
	if (sImageloader == "base") 	return E_IMAGELOADER::LOADER_BASE;
	if (sImageloader == "custom") 	return E_IMAGELOADER::LOADER_CUSTOM;
	throw std::invalid_argument("invalid imageloader conversion");
}

E_PREPROCESSOR convertPreprocessor(const std::string& sPreprocessor) {
	if (sPreprocessor == "base") 	return E_PREPROCESSOR::PREPROC_BASE;
	throw std::invalid_argument("invalid preprocessor conversion");
}

E_BGSUBTRACTOR convertBGSubtractor(const std::string& sBGSubtractor) {
	if (sBGSubtractor == "pbas") 	return E_BGSUBTRACTOR::BG_PBAS;
	if (sBGSubtractor == "fd") 	return E_BGSUBTRACTOR::BG_FD;
	throw std::invalid_argument("invalid bgsubtractor conversion");
}

E_SEGMENTATION convertSegmentation(const std::string& sSegmentation) {
	if (sSegmentation == "regiongrowing") 	return E_SEGMENTATION::E_REGIONGROWING;
	if (sSegmentation == "dbscan") 		return E_SEGMENTATION::E_DBSCAN;
	throw std::invalid_argument("invalid segmentation conversion");
}

E_POSTPROCESSOR convertPostprocessor(const std::string& sPostprocessor) {
	if (sPostprocessor == "base") 			return E_POSTPROCESSOR::POSTPROC_BASE;
	if (sPostprocessor == "interpolation")	return E_POSTPROCESSOR::POSTPROC_INTERPOLATION;
	throw std::invalid_argument("invalid postprocessor conversion");
}

E_STEREOMATCHER convertStereomatcher(const std::string& sStereomatcher) {
	if (sStereomatcher == "basicbm") 		return E_STEREOMATCHER::BASIC_BLOCK;
	if (sStereomatcher == "basicsg") 		return E_STEREOMATCHER::BASIC_SG;
	if (sStereomatcher == "basicbp") 		return E_STEREOMATCHER::BASIC_BP;
	if (sStereomatcher == "custombm")		return E_STEREOMATCHER::CUSTOM_BLOCK;
	if (sStereomatcher == "customdiff")		return E_STEREOMATCHER::CUSTOM_DIFF;
	if (sStereomatcher == "customcanny")	return E_STEREOMATCHER::CUSTOM_CANNY;
	if (sStereomatcher == "custompyramid")	return E_STEREOMATCHER::CUSTOM_PYRAMID;
	throw std::invalid_argument("invalid stereomatcher conversion");
}

std::ostream& operator << (std::ostream& os, E_IMAGELOADER eImageloader) {
	switch (eImageloader) {
	case E_IMAGELOADER::LOADER_BASE: {
		os << "base";
		break;
	}
	case E_IMAGELOADER::LOADER_CUSTOM: {
		os << "custom";
		break;
	}
	default: throw std::invalid_argument("Unknown eImageloader");
	}
	return os;
}

std::ostream& operator << (std::ostream& os, E_PREPROCESSOR ePreprocess) {
	switch (ePreprocess) {
	case E_PREPROCESSOR::PREPROC_BASE: {
		os << "base";
		break;
	}
	default: throw std::invalid_argument("Unknown ePreprocess");
	}
	return os;
}

std::ostream& operator << (std::ostream& os, E_BGSUBTRACTOR eBGSubtractor) {
	switch (eBGSubtractor) {
	case E_BGSUBTRACTOR::BG_PBAS: {
		os << "Pixel Based Adaptive Segmenter";
		break;
	}
	case E_BGSUBTRACTOR::BG_FD: {
		os << "Frame Difference";
		break;
	}
	default: throw std::invalid_argument("Unknown eBGSubtractor");
	}
	return os;
}

std::ostream& operator << (std::ostream& os, E_SEGMENTATION eSegmentation) {
	switch (eSegmentation) {
	case E_SEGMENTATION::E_REGIONGROWING: {
		os << "Region Growing";
		break;
	}
	case E_SEGMENTATION::E_DBSCAN: {
		os << "DBSCAN";
		break;
	}
	default: throw std::invalid_argument("Unknown eSegmentatoin");
	}
	return os;
}

std::ostream& operator << (std::ostream& os, E_POSTPROCESSOR ePostProcess) {
	switch (ePostProcess) {
	case E_POSTPROCESSOR::POSTPROC_BASE: {
		os << "base";
		break;
	}
	case E_POSTPROCESSOR::POSTPROC_INTERPOLATION: {
		os << "interpolation";
		break;
	}
	default: throw std::invalid_argument("Unknown ePostProcess");
	}
	return os;
}

std::ostream& operator << (std::ostream& os, E_STEREOMATCHER eStereomatch) {
	switch (eStereomatch) {
	case E_STEREOMATCHER::BASIC_BLOCK: {
		os << "Basic Block Match";
		break;
	}
	case E_STEREOMATCHER::BASIC_SG: {
		os << "Basic Semi Global Match";
		break;
	}
	case E_STEREOMATCHER::BASIC_BP: {
		os << "Basic Belief Propagation";
		break;
	}
	case E_STEREOMATCHER::CUSTOM_BLOCK: {
		os << "Custom Block Match";
		break;
	}
	case E_STEREOMATCHER::CUSTOM_DIFF: {
		os << "Custom Differential Match";
		break;
	}
	case E_STEREOMATCHER::CUSTOM_CANNY: {
		os << "Custom Canny Match";
		break;
	}
	case E_STEREOMATCHER::CUSTOM_PYRAMID: {
		os << "Custom Pyramid Match";
		break;
	}
	default: throw std::invalid_argument("Unknown eStereomatch");
	}
	return os;
}
