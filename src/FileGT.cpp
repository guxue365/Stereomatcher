#include <FileGT.h>

#include <iostream>
#include <fstream>

#include <nlohmann/json.hpp>


using namespace std;
using json = nlohmann::json;

FileGT::FileGT() : msFilename("data_gt.json") {

}

FileGT::FileGT(const std::string& sFile) : msFilename(sFile) {

}

FileGT::~FileGT() {

}

void FileGT::AddFrameGT(const FrameGT& rFrameGT) {
	maFrameGT = LoadFramesFromFile(msFilename);

	maFrameGT.push_back(rFrameGT);
	maFrameGT.back().miID = static_cast<int>(maFrameGT.size()) - 1;

	cout << "Added new GT: " << maFrameGT.back().miID;
	cout << " - Frame: " << rFrameGT.miFrame << endl;
	cout << " - Label: " << rFrameGT.miLabel << endl;
	cout << " - Position: " << rFrameGT.mdX << " | " << rFrameGT.mdY << " | " << rFrameGT.mdZ << endl;

	SaveFramesToFile(msFilename, maFrameGT);
}

std::vector<FrameGT> FileGT::LoadFramesFromFile(const std::string& sFile) {
	ifstream oFile(sFile.c_str(), ios::in);
	if (!oFile.is_open()) {
		cout << "Warning: Specified GT File is empty. Will create new File" << endl;
		oFile.close();
		return vector<FrameGT>();
	}

	json oJsonFile;

	oFile >> oJsonFile;

	oFile.close();

	vector<FrameGT> aResult;

	if (oJsonFile["frames"].empty() || !oJsonFile["frames"].is_array()) {
		throw std::invalid_argument("GT File is invalid");
	}
	for (json oJsonFrame : oJsonFile["frames"]) {
		FrameGT oFrame;

		oFrame.miID = oJsonFrame["id"];
		oFrame.miFrame = oJsonFrame["frame"];
		oFrame.miLabel = oJsonFrame["label"];
		vector<double> aPosition = oJsonFrame["position"];
		oFrame.mdX = aPosition[0];
		oFrame.mdY = aPosition[1];
		oFrame.mdZ = aPosition[2];

		if (oFrame.miID != aResult.size()) {
			throw std::invalid_argument("GT Frame have invalid ID");
		}

		aResult.push_back(oFrame);
	}

	return aResult;
}

void FileGT::SaveFramesToFile(const std::string& sFile, const std::vector<FrameGT>& aFrames) {
	json oJsonResult;
	json aJsonFrames;

	for (auto& oFrame : aFrames) {
		json oJsonFrame;
		oJsonFrame["id"] = oFrame.miID;
		oJsonFrame["frame"] = oFrame.miFrame;
		oJsonFrame["label"] = oFrame.miLabel;
		oJsonFrame["position"] = { oFrame.mdX, oFrame.mdY, oFrame.mdZ };

		aJsonFrames.push_back(oJsonFrame);
	}
	oJsonResult["frames"] = aJsonFrames;

	ofstream oJsonFile(sFile.c_str(), ios::out);
	if (!oJsonFile.is_open()) {
		throw std::invalid_argument("Cannot save GT File");
	}

	oJsonFile << oJsonResult.dump(5);

	oJsonFile.close();
}