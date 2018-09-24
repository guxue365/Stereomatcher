#include <vector>
#include <string>


struct FrameGT {
	int miID;
	int miFrame;
	int miLabel;
	double mdX;
	double mdY;
	double mdZ;
};

class FileGT {
public:
	FileGT();
	FileGT(const std::string& sFile);
	virtual ~FileGT();

	void AddFrameGT(const FrameGT& rFrameGT);
private:
	std::vector<FrameGT> maFrameGT;
	std::string msFilename;

	std::vector<FrameGT> LoadFramesFromFile(const std::string& sFile);
	void SaveFramesToFile(const std::string& sFile, const std::vector<FrameGT>& aFrames);
};