#include <vector>
#include <cstdlib>
#include <iostream>

struct bounding_box
{
  float2 lower_left;
  float2 upper_right;

  bounding_box()
  {
    // initialize an empty box
    lower_left  = make_float2( 1e12f,  1e12f);
    upper_right = make_float2(-1e12f, -1e12f);
  }

  // construct a box from a single point
  bounding_box(float2 p)
  {
    lower_left  = p;
    upper_right = p;
  }

  // construct a box from two corners
  bounding_box(float2 ll, float2 ur)
  {
    lower_left = ll;
    upper_right = ur;
  }

  // return the center of the box
  float2 center() const
  {
    return make_float2(0.5f * (lower_left.x + upper_right.x), 0.5f * (lower_left.y + upper_right.y));
  }

  // add that box to this one and return the result
  bounding_box operator+(const bounding_box &box)
  {
    float min_x = min(lower_left.x, box.lower_left.x);
    float min_y = min(lower_left.y, box.lower_left.y);
    float max_x = max(upper_right.x, box.upper_right.x);
    float max_y = max(upper_right.y, box.upper_right.y);

    return bounding_box(make_float2(min_x, min_y), make_float2(max_x, max_y));
  }
};


void generate_random_points(std::vector<float2> &points)
{
  // sequentially generate some random 2D points in the unit square
  // TODO parallelize this loop using thrust::tabulate
  for(int i = 0; i < points.size(); ++i)
  {
    float x = float(rand()) / RAND_MAX;
    float y = float(rand()) / RAND_MAX;

    points[i] = make_float2(x,y);
  }
}


bounding_box compute_bounding_box(const std::vector<float2> &points)
{
  // start with an empty box
  bounding_box result;

  // sequentially increase the size of result to include each point
  // TODO parallelize this loop using thrust::reduce
  for(int i = 0; i < points.size(); ++i)
  {
    // create a bounding box containing only the current point
    bounding_box current_point = bounding_box(points[i]);

    // increase the size of the box
    result = result + current_point;
  }

  return result;
}


void classify(const std::vector<float2> &points, float2 center, std::vector<int> &quadrants)
{
  // classify each point as 
  // TODO parallelize this loop using thrust::transform
  for(int i = 0; i < points.size(); ++i)
  {
    float x = points[i].x;
    float y = points[i].y;

    // bottom-left:  0
    // bottom-right: 1
    // top-left:     2
    // top-right:    3

    quadrants[i] = (x <= center.x ? 0 : 1) | (y <= center.y ? 0 : 2);
  }
}


void count_points_in_quadrants(std::vector<float2> &points, const std::vector<int> &quadrants, std::vector<int> &counts_per_quadrant)
{
  // sequentially compute a histogram
  // TODO parallelize this operation by
  //   1. sorting points by quadrant
  //   2. reducing points by quadrant
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

  std::vector<float2> points(num_points);

  generate_random_points(points);

  for(int i = 0, n = points.size() ; i < n ; ++i)
    std::cout << "points[" << i << "]: x=" << points[i].x << ", y=" << points[i].y << std::endl;
  std::cout << std::endl;

  bounding_box box = compute_bounding_box(points);
  float2 center = box.center();

  std::cout << "Bounding box: lower_left=(" << box.lower_left.x << ", " << box.lower_left.y << ")" << std::endl;
  std::cout <<              " lower_right=(" << box.upper_right.x << ", " << box.upper_right.y << ")" << std::endl;
  std::cout <<              " center=(" << center.x << ", " << center.y << ")" << std::endl;
  std::cout << std::endl;

  std::vector<int> quadrants(points.size());
  classify(points, center, quadrants);

  std::vector<int> counts_per_quadrant(4);
  count_points_in_quadrants(points, quadrants, counts_per_quadrant);

  std::cout << "Per-quadrant counts:" << std::endl;
  std::cout << "  Bottom-left : " << counts_per_quadrant[0] << " points" << std::endl;
  std::cout << "  Bottom-right: " << counts_per_quadrant[1] << " points" << std::endl;
  std::cout << "  Top-left    : " << counts_per_quadrant[2] << " points" << std::endl;
  std::cout << "  Top-right   : " << counts_per_quadrant[3] << " points" << std::endl;
  std::cout << std::endl;

  for(int i = 0, n = points.size() ; i < n ; ++i)
    std::cout << "points[" << i << "]: x=" << points[i].x << ", y=" << points[i].y << std::endl;
  std::cout << std::endl;
}

