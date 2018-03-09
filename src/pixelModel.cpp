/*
 * pixelModel.cpp
 *
 *  Created on: Mar 2, 2018
 *      Author: vinay
 */

#include "pixelModel.h"

pixelModel::pixelModel(int maxGauss) {
	CurrentGaussians = 0;

	for (int i=0; i < maxGauss; i++){
		Vec3d dummy = {0,0,0};
		means.push_back(dummy);
		wts.push_back(0);
		vars.push_back(0);
		}
}

pixelModel::~pixelModel() {}
