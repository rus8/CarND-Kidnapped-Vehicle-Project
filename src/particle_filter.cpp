/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 7;

	default_random_engine gen;
	// Create normal distributions to sample particles
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i =0; i < num_particles; i++){
		Particle sample;
		sample.id = i;

		sample.x = dist_x(gen);
		sample.y = dist_y(gen);
		sample.theta = dist_theta(gen);

		sample.weight = 1;
		weights.push_back(sample.weight);

		particles.push_back(sample);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine gen;
    // Create normal distributions to sample particles
    normal_distribution<double> dist_x(0.0, std_pos[0]);
    normal_distribution<double> dist_y(0.0, std_pos[1]);
    normal_distribution<double> dist_theta(0.0, std_pos[2]);

    for (int i =0; i < num_particles; i++) {
        if (fabs(yaw_rate) < 0.0001) {
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
        } else {
            particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) -
                                sin(particles[i].theta)) + dist_x(gen);
            particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) -
                                cos(particles[i].theta + yaw_rate * delta_t)) + dist_y(gen);
            particles[i].theta = particles[i].theta + yaw_rate * delta_t + dist_theta(gen);
        }
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations, double sensor_range) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    double distance;
    double min_dist = 2 * sensor_range;
    for (auto& observ : observations){
        for (auto prediction : predicted){
            distance = dist(prediction.x, prediction.y, observ.x, observ.y);
            if (distance < min_dist){
                min_dist = distance;
                observ.id = prediction.id;
            }
        }
        min_dist = 2 * sensor_range;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    //Some common variables
    //for landmarks selection
    double distance;
    //for transformation
    double x;
    double y;
    //for assosiation
    int id;
    //for probability calculation
    double coef = 0.5 / M_PI / std_landmark[0] / std_landmark[1];
    double expon;
    double prob;
    double var_x = std_landmark[0]*std_landmark[0];
    double var_y = std_landmark[1]*std_landmark[1];

    weights.clear();

    for (int i = 0; i < num_particles; i++){
        std::vector<LandmarkObs> close_landmarks;
        //select possible landmarks
        for (auto landmark : map_landmarks.landmark_list){
            distance = dist(particles[i].x, particles[i].y, landmark.x_f, landmark.y_f);
            if (distance < sensor_range){
                close_landmarks.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
            }
        }

        //transform observations
        std::vector<LandmarkObs> transformed_observations;
        /*for (auto& observ : observations){
            x = observ.x;
            y = observ.y;
            observ.x = particles[i].x + cos(particles[i].theta) * x -
                                        sin(particles[i].theta) * y;
            observ.y = particles[i].y + sin(particles[i].theta) * x +
                                        cos(particles[i].theta) * y;
        }*/

        for (auto observ: observations) {
            x = particles[i].x + cos(particles[i].theta) * observ.x -
                       sin(particles[i].theta) * observ.y;
            y = particles[i].y + sin(particles[i].theta) * observ.x +
                       cos(particles[i].theta) * observ.y;

            double minDist = std::numeric_limits<float>::max();
            for (auto landmark: close_landmarks) {
                distance = dist(x, y, landmark.x, landmark.y);
                if (distance < minDist) {
                    minDist = distance;
                    id = landmark.id;
                }
            }
            transformed_observations.push_back(LandmarkObs{id, x, y});
        }

//        dataAssociation(close_landmarks, observations, sensor_range);
        //weights calculation
        particles[i].weight = 1.0;
        for (auto observ : transformed_observations){
            for (auto landmark : close_landmarks){
                if (observ.id == landmark.id) {
                    expon = (-0.5) * ((observ.x - landmark.x) * (observ.x - landmark.x) / var_x +
                                    (observ.y - landmark.y) * (observ.y - landmark.y) / var_y);
                    prob = coef * exp(expon);
                    particles[i].weight *= prob;
                    break;
                }
            }
        }
        weights.push_back(particles[i].weight);
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::vector<Particle> new_particles;

    default_random_engine gen;
    std::discrete_distribution<> dicr_dist(weights.begin(), weights.end());

    for (int i = 0; i < num_particles; i++) {
        new_particles.push_back(particles[dicr_dist(gen)]);
    }

    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
