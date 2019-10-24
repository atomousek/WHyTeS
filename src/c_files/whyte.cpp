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
    /*
    calculates the probability of the occurrence of the tested vector using model
    input:
        tested vector (time, x, y, heading, speed)
        model: 
            no_clusters .. number of clusters, 
            no_periods .. number of chosen periodicities to build hypertime, 
            periodicities .. set of the most influencing periodicities,
            C .. set of cluster centers
            W .. set of cluster weights
            PREC .. set of precision matices (inversed covariance matrices)
    */
    long spatial_dim = 4;  // pos_x, pos_y, vel_x, vel_y
    double degrees = spatial_dim + 2 * no_periods; // degrees of freedom for chi2
    double distance; //mahalanobis distance between each C and projection
    double tmp; // subtotal during distance calculation
    double prob = 0.0; // "probability" of the occurrence of tested position

    double projection[spatial_dim + 2 * no_periods]; //projection of tested vector into warped hypertime space
    double shifted[spatial_dim + 2 * no_periods]; // temporal variable, projection minus centre
    /* filling the projection*/
    projection[0] = x;
    projection[1] = y;
    projection[2] = cos(heading) * speed; // velocity in the direction of x
    projection[3] = sin(heading) * speed; // velocity in the direction of y

    for(int id_n_p = 0; id_n_p < no_periods; id_n_p++)
    {
        projection[spatial_dim + 2 * id_n_p] = cos(time * 2.0 * M_PI / periodicities[id_n_p]);
        projection[spatial_dim + 2 * id_n_p + 1] = sin(time * 2.0 * M_PI / periodicities[id_n_p]);
    }
    /* calculate mahalanobis distance between projection and every cluster centre*/
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
        /*probability of occurrence from the point of view of every cluster (distribution estimation)*/
        prob += gsl_cdf_chisq_Q(distance, (double) degrees) * W[c];
    }
    /*return sum of particular and weighted probabilities*/
    return prob;
}



