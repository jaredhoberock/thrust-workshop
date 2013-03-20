#include <thrust/device_vector.h>
#include <thrust/tabulate.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <cstdlib>
#include <iostream>
#include <cassert>
#include "time_invocation_cuda.hpp"


__host__ __device__
float2 operator+(float2 a, float2 b)
{
  return make_float2(a.x + b.x, a.y + b.y);
}


// given an integer, output a pseudorandom 2D point
struct random_point
{
  __host__ __device__ unsigned int hash(unsigned int x)
  {
    x = (x+0x7ed55d16) + (x<<12);
    x = (x^0xc761c23c) ^ (x>>19);
    x = (x+0x165667b1) + (x<<5);
    x = (x+0xd3a2646c) ^ (x<<9);
    x = (x+0xfd7046c5) + (x<<3);
    x = (x^0xb55a4f09) ^ (x>>16);
    return x;
  }

  __host__ __device__
  float2 operator()(unsigned int x)
  {
    return make_float2(float(hash(x)) / UINT_MAX, float(hash(2 * x)) / UINT_MAX);
  }
};


void generate_random_points(thrust::device_vector<float2> &points)
{
  thrust::tabulate(points.begin(), points.end(), random_point());
}


float2 compute_centroid(const thrust::device_vector<float2> &points)
{
  float2 init = make_float2(0,0);

  // compute the sum
  float2 sum = thrust::reduce(points.begin(), points.end(), init); 
  // divide the sum by the number of points
  return make_float2(sum.x / points.size(), sum.y / points.size());
}


// given a 2D point, return which quadrant it is in
struct classify_point
{
  float2 center;

  __host__ __device__
  classify_point(float2 c)
  {
    center = c;
  }

  __host__ __device__
  unsigned int operator()(float2 p)
  {
    return (p.x <= center.x ? 0 : 1) | (p.y <= center.y ? 0 : 2);
  }
};


void classify_points_by_quadrant(const thrust::device_vector<float2> &points, float2 center, thrust::device_vector<int> &quadrants)
{
  // classify each point relative to the centroid
  thrust::transform(points.begin(), points.end(), quadrants.begin(), classify_point(center));
}


void count_points_in_quadrants(thrust::device_vector<float2> &points, thrust::device_vector<int> &quadrants, thrust::device_vector<int> &counts_per_quadrant)
{
  // sort points by quadrant
  thrust::sort_by_key(quadrants.begin(), quadrants.end(), points.begin());

  // count points in each quadrant
  thrust::reduce_by_key(quadrants.begin(), quadrants.end(),
                        thrust::constant_iterator<int>(1),
                        thrust::discard_iterator<>(),
                        counts_per_quadrant.begin());
}


// pass device_vector by pointer so that time_invocation_cuda doesn't
// try to make copies of them
void run_experiment(thrust::device_vector<float2> *points,
                    thrust::device_vector<int> *quadrants,
                    thrust::device_vector<int> *counts_per_quadrant)
{
  generate_random_points(*points);

  float2 centroid = compute_centroid(*points);

  classify_points_by_quadrant(*points, centroid, *quadrants);

  count_points_in_quadrants(*points, *quadrants, *counts_per_quadrant);
}


int main()
{
  const size_t num_points = 10000000;

  thrust::device_vector<float2> points(num_points);
  thrust::device_vector<int> quadrants(num_points);
  thrust::device_vector<int> counts_per_quadrant(4);

  std::cout << "Warming up...\n" << std::endl;

  // validate and warm up the JIT
  run_experiment(&points, &quadrants, &counts_per_quadrant);

  int expected_count_per_quadrant = num_points / 4;
  int tolerance = expected_count_per_quadrant / 1000;
  for(int i = 0; i < 4; ++i)
  {
    assert(abs(expected_count_per_quadrant - counts_per_quadrant[i]) < tolerance);
  }

  std::cout << "Timing...\n" << std::endl;

  size_t num_trials = 25;
  double mean_msecs = time_invocation_cuda(num_trials, run_experiment, &points, &quadrants, &counts_per_quadrant);
  double mean_secs = mean_msecs / 1000;
  double millions_of_points = double(num_points) / 1000000;

  std::cout << millions_of_points / mean_secs << " millions of points generated and counted per second." << std::endl;
}

