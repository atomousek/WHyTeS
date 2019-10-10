#include <cmath>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>
#include "whyte.h"

WHyTe::WHyTe(){}

WHyTe::~WHyTe(){}

void WHyTe::readFromXML(const std::string &fileName)
{

}

double WHyTe::getLikelihood(double time, double x, double y, double heading, double speed)
{
    double distance;
    double tmp;
    double spatial_dim = 4;  // pos_x, pos_y, vel_x, vel_y
    double degrees = spatial_dim + 2 * no_periods;
    double prob = 0.0;

    /* projection len = 4 + 2 * no_periods */
    double projection[spatial_dim + 2 * no_periods];
    double shifted[spatial_dim + 2 * no_periods];
    projection[0] = x;
    projection[1] = y;
    projection[2] = cos(heading) * speed;
    projection[3] = sin(heading) * speed;

    for(int id_n_p = 0; id_n_p < no_periods; id_n_p++)
    {
        projection[spatial_dim + 2 * id_n_p] = cos(time * 2.0 * M_PI / periodicities[id_n_p]);
        projection[spatial_dim + 2 * id_n_p + 1] = sin(time * 2.0 * M_PI / periodicities[id_n_p]);
    }
    
    for(int c = 0; c < no_clusters; c++)
    {
        for(int i = 0; i < degrees; i++)
        {
            shifted[i] = projection[i] - C[c][i];
        }
        distance = 0.0;
        for(int j = 0; j < degrees; j++)
        {
            tmp = 0.0;
            for(int i = 0; i < degrees; i++)
            {
                tmp = tmp + PREC[c][i][j] * shifted[i];
            }
            tmp = tmp * shifted[j];
            distance += tmp;
        }
        prob += gsl_cdf_chisq_Q(distance, (double) degrees) * W[c];
    }

    return prob;
}



