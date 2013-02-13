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

// an axis-aligned 2D bounding_box is a lower left corner and an upper right corner
struct bounding_box
{
  float2 lower_left;
  float2 upper_right;

  // construct an empty box
  __host__ __device__
  bounding_box()
  {
    // initialize an empty box
    lower_left  = make_float2( 1e12f,  1e12f);
    upper_right = make_float2(-1e12f, -1e12f);
  }

  // construct a box from a single point
  __host__ __device__
  bounding_box(float2 p)
  {
    lower_left  = p;
    upper_right = p;
  }

  // construct a box from two corners
  __host__ __device__
  bounding_box(float2 ll, float2 ur)
  {
    lower_left = ll;
    upper_right = ur;
  }

  // return the center of the box
  __host__ __device__
  float2 center() const
  {
    return make_float2(0.5f * (lower_left.x + upper_right.x), 0.5f * (lower_left.y + upper_right.y));
  }

  // add that box to this one and return the result
  __host__ __device__
  bounding_box operator+(const bounding_box &box) const
  {
    float min_x = min(lower_left.x, box.lower_left.x);
    float min_y = min(lower_left.y, box.lower_left.y);
    float max_x = max(upper_right.x, box.upper_right.x);
    float max_y = max(upper_right.y, box.upper_right.y);

    return bounding_box(make_float2(min_x, min_y), make_float2(max_x, max_y));
  }
};


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


bounding_box compute_bounding_box(const thrust::device_vector<float2> &points)
{
  bounding_box empty;
  return thrust::reduce(points.begin(), points.end(), empty);
}


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


void classify(const thrust::device_vector<float2> &points, float2 center, thrust::device_vector<int> &quadrants)
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
  const size_t num_points = 10;

  thrust::device_vector<float2> points(num_points);

  generate_random_points(points);

  for(int i = 0; i < points.size(); ++i)
    std::cout << "points[" << i << "]: " << points[i] << std::endl;
  std::cout << std::endl;

  bounding_box box = compute_bounding_box(points);
  float2 center = box.center();

  std::cout << "Bounding box:" << std::endl;
  std::cout << "  lower_left:  " << box.lower_left << std::endl;
  std::cout << "  lower_right: " << box.upper_right << std::endl;
  std::cout << "  center:      " << center << std::endl;
  std::cout << std::endl;

  thrust::device_vector<int> quadrants(points.size());
  classify(points, center, quadrants);

  std::cout << "Quadrants: " << std::endl;
  for(int i = 0; i < quadrants.size(); ++i)
    std::cout << "  " << i << ": " << quadrants[i] << std::endl;

  thrust::device_vector<int> counts_per_quadrant(4);
  count_points_in_quadrants(points, quadrants, counts_per_quadrant);

  std::cout << "Per-quadrant counts:" << std::endl;
  std::cout << "  Bottom-left : " << counts_per_quadrant[0] << " points" << std::endl;
  std::cout << "  Bottom-right: " << counts_per_quadrant[1] << " points" << std::endl;
  std::cout << "  Top-left    : " << counts_per_quadrant[2] << " points" << std::endl;
  std::cout << "  Top-right   : " << counts_per_quadrant[3] << " points" << std::endl;
  std::cout << std::endl;

  for(int i = 0; i < points.size(); ++i)
    std::cout << "points[" << i << "]: " << points[i] << std::endl;
  std::cout << std::endl;
}

