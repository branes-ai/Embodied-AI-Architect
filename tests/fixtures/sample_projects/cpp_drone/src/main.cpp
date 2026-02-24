#include <iostream>
#include "perception.h"
#include "control.h"

int main(int argc, char** argv) {
    PerceptionPipeline perception;
    ControlLoop control;

    while (true) {
        auto detections = perception.process_frame();
        control.update(detections);
    }

    return 0;
}
