/*
 * pixelModel.h
 *
 *  Created on: Mar 2, 2018
 *      Author: vinay
 */

#ifndef PIXELMODEL_H_
#define PIXELMODEL_H_

#include <opencv2/core/core.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

class pixelModel {
public:
	pixelModel(int);
	virtual ~pixelModel();

	int CurrentGaussians;			// current number of gaussians present in the model
	vector<Vec3d> means;			// means related to different gaussians
	vector<double> vars;				// variance of different gaussians
	vector<double> wts;				// weights of different gaussians
};

#endif /* PIXELMODEL_H_ */
