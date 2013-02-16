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
#include <cstdio>


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
    thrust::default_random_engine rng(hash(x));
    thrust::random::uniform_real_distribution<float> dist;
    return make_float2(dist(rng), dist(rng));
  }
};


void generate_random_points(thrust::device_vector<float2> &points)
{
  thrust::tabulate(points.begin(), points.end(), random_point());
}


float2 compute_centroid(const thrust::device_vector<float2> &points)
{
  float2 init = make_float2(0,0);
  float2 sum = thrust::reduce(points.begin(), points.end(), init); 
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


std::ostream &operator<<(std::ostream &os, float2 p)
{
  return os << "(" << p.x << ", " << p.y << ")";
}


int main()
{
  const size_t num_points = 10000000;

  thrust::device_vector<float2> points(num_points);

  generate_random_points(points);

  float2 centroid = compute_centroid(points);

  thrust::device_vector<int> quadrants(points.size());
  classify_points_by_quadrant(points, centroid, quadrants);

  thrust::device_vector<int> counts_per_quadrant(4);
  count_points_in_quadrants(points, quadrants, counts_per_quadrant);

  std::cout << "Per-quadrant counts:" << std::endl;
  std::cout << "  Bottom-left : " << counts_per_quadrant[0] << " points" << std::endl;
  std::cout << "  Bottom-right: " << counts_per_quadrant[1] << " points" << std::endl;
  std::cout << "  Top-left    : " << counts_per_quadrant[2] << " points" << std::endl;
  std::cout << "  Top-right   : " << counts_per_quadrant[3] << " points" << std::endl;
  std::cout << std::endl;
}

