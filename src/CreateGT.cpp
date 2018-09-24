#include <iostream>
#include <vector>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <nlohmann/json.hpp>

#include <FileGT.h>

using namespace std;
using namespace cv;
using json = nlohmann::json;

void onMouse(int event, int x, int y, int, void*) {
	if (event != EVENT_LBUTTONDOWN)		return;

	cout << "Mouse Event at: " << x << " | " << y << endl;
}



int main() {

	FileGT file("gt.json");

	file.AddFrameGT({ 2, 3, 4, 2.0, 3.0, 4.0 });

	return 0;

	Mat oImage = imread("E:/sample_images/img_0.png", IMREAD_GRAYSCALE);
	oImage *= 3;
	applyColorMap(oImage, oImage, COLORMAP_JET);
	imshow("Disp", oImage);

	setMouseCallback("Disp", onMouse);

	bool bRunning = true;
	for (int iFrame = 0; bRunning; ++iFrame) {
		int iKeyCode = waitKey(0);
		if (iKeyCode == 27)	break;

		switch (iKeyCode) {
		case 49: {
			cout << "Setting Label 1" << endl;
			break;
		}
		case 50: {
			cout << "Setting Label 2" << endl;
			break;
		}
		}
	}

	return 0;
}