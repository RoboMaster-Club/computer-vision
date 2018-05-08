//
// Created by why on 5/8/18.
//

#ifndef ARMOR_DETECTION_ARMOR_H
#define ARMOR_DETECTION_ARMOR_H

#include "SearchArea.h"

class Armor {
public:
    int id;
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

    float clacScore();
    SearchArea getSearchArea();
};


#endif //ARMOR_DETECTION_ARMOR_H
