#include <stdio.h>
#include "whyte.h"

// ./main < test_dataset_to_test.txt > out.txt

int main(int argc, char *argv[])
{
    WHyTe whyte;

    whyte.readFromXML("whyte_map.xml");

    double time;
    double x;
    double y;
    double heading;
    double speed;
    double prob;
    int ret;
    while(true)
    {
        //printf("time x y heading speed\n");
        ret = scanf("%lf %lf %lf %lf %lf\n", &time, &x, &y, &heading, &speed);
        if(ret == 5)
        {
            prob = whyte.getLikelihood(time, x, y, heading, speed);
            //printf("prob: %f\n\n", prob);
            printf("%.20f\n", prob);
        }
        else
        {
            break;
        }
    }

    return 0;
}
