// ProceduralEnvironments.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "Engine.h"

int main()
{
    Engine* engine = new Engine();
    engine->run();
    delete(engine);

    return 0;
}