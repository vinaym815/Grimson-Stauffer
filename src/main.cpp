#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>		//Also included in class pixelModel
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include "pixelModel.h"

using namespace cv;
using namespace std;

double alpha = 0.001;
double initVar = 255;
double initWeight = 0.001;
double wtThreshold = 0.8;

void updatePixel(Vec3d &, pixelModel &, const int &);
double GaussianPdf(const Vec3d &dis,const double &var);
vector<int> argS(const vector<double> &);

int main(int argc, char *argv[]){
	string filename;
	int MaxGaussians;

	filename = "../umcp.mpg";
	if (argc > 2 && argc <3){
		filename = argv[1];
	}
    else if (argc > 3){
		filename = argv[1];
		MaxGaussians = atoi(argv[2]);
    }
    else{
	MaxGaussians = 5;
	filename = "../umcp.mpg";
    }

	VideoCapture cap(filename);
	if ( !cap.isOpened() ) {
		cout << "Cannot open the video file. \n";
		return -1;
	}

	int frameWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int frameHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    VideoWriter video("output.avi",CV_FOURCC('M','J','P','G'),10, 
            Size(frameWidth,frameHeight),true);
	Mat frame, editFrame;

	// Starting frame Struture consisting of pixels models
	vector<vector <pixelModel>> frameStrut;
	for (int i=0; i < frameHeight; i++){
		vector<pixelModel> dummy;
		for (int j=0; j < frameWidth; j++)
			dummy.push_back(pixelModel(MaxGaussians));
		frameStrut.push_back(dummy);
	}

	namedWindow("Original_Video",CV_WINDOW_AUTOSIZE);
	namedWindow("Edited_Video",CV_WINDOW_AUTOSIZE);

	while(true) {
		if (!cap.read(frame))
			break;

		editFrame = frame.clone();
        // Converting the image from CV_8U to CV_64 because CV_8U
        // can not have negative values which leads to error in 
        // computing distance for an Intensity values lying on left 
        // side of an gaussian
		editFrame.convertTo(editFrame,CV_64F);

        // for trying to imporve performance 
		#pragma omp parallel for
		for(int i = 0; i < frame.rows; i++)
			for (int j=0; j<frame.cols; j++)
				updatePixel(editFrame.at<Vec3d>(i,j), frameStrut[i][j], MaxGaussians);

		editFrame.convertTo(editFrame,CV_8U);
        video.write(editFrame);

		imshow("Original_Video", frame);
		imshow("Edited_Video", editFrame);

        // Detecting if key 'q' is pressed to make an exit
		if(waitKey(1) == 113)
			break;
	}
    video.release();
	cout << "Finish !!!!" << endl;
	return 0;
}


void updatePixel(Vec3d &currentPixel, pixelModel &frameStrut, const int &MaxGaussians){

	bool falgModelFit = false;
	for(int i =0; i < frameStrut.CurrentGaussians; i++){
		Vec3d dis = currentPixel - frameStrut.means[i];

		if (norm(dis) < 2.5*sqrt(frameStrut.vars[i])){
			falgModelFit = true;
			double rho  = GaussianPdf(dis, frameStrut.vars[i]);
			frameStrut.wts[i] = (1-alpha)*frameStrut.wts[i] + alpha;
			frameStrut.means[i] = (1-rho)*frameStrut.means[i] + rho*currentPixel;
			frameStrut.vars[i] = (1-rho)*frameStrut.vars[i] + rho*pow(norm(dis),2);
			break;
		}
		else
			frameStrut.wts[i] = (1-alpha)*frameStrut.wts[i];
	}

	if (!falgModelFit){
		int ind = 0;
		if (frameStrut.CurrentGaussians < MaxGaussians){
			ind = frameStrut.CurrentGaussians;
			frameStrut.CurrentGaussians += 1;
		}
		else{
			vector <double> wtbyvar;
			for (int i = 0; i < MaxGaussians; i++)
				wtbyvar.push_back(frameStrut.wts[i]/sqrt(frameStrut.vars[i]));

			vector<int> sortInd = argS(wtbyvar);
			ind = sortInd[0];
		}
		frameStrut.wts[ind] = initWeight;
		frameStrut.means[ind] = currentPixel;
		frameStrut.vars[ind] = initVar;
	}

	// normalizing the weights
	double wtSum = 0;
	for (int i = 0; i < frameStrut.CurrentGaussians; i++)
		wtSum += frameStrut.wts[i];

	for (int i = 0; i < frameStrut.CurrentGaussians; i++)
		frameStrut.wts[i] = frameStrut.wts[i]/wtSum;

	// sorting based upon weight/variance
	vector <double> wtbyvar1;
	for (int i = 0; i < frameStrut.CurrentGaussians; i++)
		wtbyvar1.push_back(frameStrut.wts[i]/sqrt(frameStrut.vars[i]));

	vector<int> sortInd = argS(wtbyvar1);

	wtSum = 0;
	for (int i =0; i < frameStrut.CurrentGaussians; i++){
		int ind = sortInd[frameStrut.CurrentGaussians - i - 1];
		Vec3d dis = currentPixel - frameStrut.means[ind];
		if (norm(dis) < 2.5*sqrt(frameStrut.vars[ind]))
			currentPixel = {255, 255, 255};
		wtSum += frameStrut.wts[ind];
		if (wtSum > wtThreshold)
			break;
	}
}

vector<int> argS(const vector<double> &x){
    vector<int> y(x.size());
    size_t n(0);
    generate(begin(y), end(y), [&]{ return n++; });
    sort(begin(y), end(y), [&](int i1, int i2) { return x[i1] < x[i2]; } );
    return y;
}

double GaussianPdf(const Vec3d &dis,const double &var){
	double pdf = 0;
	pdf = 1/sqrt(pow(2*M_PI*var,3)) * exp(-1/(2*var) * pow(norm(dis),2));
	return pdf;
}
