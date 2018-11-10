#ifndef ARMOR_DETECTION_ARMOR_H
#define ARMOR_DETECTION_ARMOR_H

#include <opencv2/opencv.hpp>
//#include "SearchArea.h"

typedef struct _Armor {
public:
    float width;
    float height;
    float x;
    float y;
    float z; //distance
    float internal_velocity_x;//pixels per frame
    float internal_velocity_y;
    float angular_velocity_x;//rad/s
    float angular_velocity_y;
    float velocity_z;

//    Armor(cv::Rect);
//
//    int clacScore();
//    cv:Rect getSearchArea();
} Armor;


#endif //ARMOR_DETECTION_ARMOR_H
