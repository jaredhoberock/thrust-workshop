#include <vector>
#include <cstdlib>
#include <iostream>


std::ostream &operator<<(std::ostream &os, float2 p)
{
  return os << "(" << p.x << ", " << p.y << ")";
}


// TODO: annotate this function with __host__ __device__ so
//       so that they are able to work with Thrust
float2 operator+(float2 a, float2 b)
{
  return make_float2(a.x + b.x, a.y + b.y);
}


void generate_random_points(std::vector<float2> &points)
{
  // sequentially generate some random 2D points in the unit square
  std::cout << "TODO: parallelize this loop using thrust::tabulate\n" << std::endl;

  for(int i = 0; i < points.size(); ++i)
  {
    float x = float(rand()) / RAND_MAX;
    float y = float(rand()) / RAND_MAX;

    points[i] = make_float2(x,y);
  }
}


float2 compute_centroid(const std::vector<float2> &points)
{
  float2 sum = make_float2(0,0);

  // compute the sum
  std::cout << "TODO: parallelize this sum using thrust::reduce\n" << std::endl;
  for(int i = 0; i < points.size(); ++i)
  {
    sum = sum + points[i];
  }

  // divide the sum by the number of points
  return make_float2(sum.x / points.size(), sum.y / points.size());
}


void classify_points_by_quadrant(const std::vector<float2> &points, float2 centroid, std::vector<int> &quadrants)
{
  // classify each point relative to the centroid
  std::cout << "TODO: parallelize this loop using thrust::transform\n" << std::endl;
  for(int i = 0; i < points.size(); ++i)
  {
    float x = points[i].x;
    float y = points[i].y;

    // bottom-left:  0
    // bottom-right: 1
    // top-left:     2
    // top-right:    3

    quadrants[i] = (x <= centroid.x ? 0 : 1) | (y <= centroid.y ? 0 : 2);
  }
}


void count_points_in_quadrants(std::vector<float2> &points, std::vector<int> &quadrants, std::vector<int> &counts_per_quadrant)
{
  // sequentially compute a histogram
  std::cout << "TODO: parallelize this loop by" << std::endl;
  std::cout << "   1. sorting points by quadrant" << std::endl;
  std::cout << "   2. reducing points by quadrant\n" << std::endl;
  for(int i = 0; i < quadrants.size(); ++i)
  {
    int q = quadrants[i];

    // increment the number of points in this quadrant
    counts_per_quadrant[q]++;
  }
}


int main()
{
  const size_t num_points = 10;

  std::cout << "TODO: move these points to the GPU using thrust::device_vector\n" << std::endl;
  std::vector<float2> points(num_points);

  generate_random_points(points);

  for(int i = 0; i < points.size(); ++i)
    std::cout << "points[" << i << "]: " << points[i] << std::endl;
  std::cout << std::endl;

  float2 centroid = compute_centroid(points);

  std::cout << "TODO: move these quadrants to the GPU using thrust::device_vector\n" << std::endl;
  std::vector<int> quadrants(points.size());
  classify_points_by_quadrant(points, centroid, quadrants);

  std::cout << "TODO: move these counts to the GPU using thrust::device_vector\n" << std::endl;
  std::vector<int> counts_per_quadrant(4);
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

