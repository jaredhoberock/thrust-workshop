Fun with Points!
================

In this exercise, we'll become familiar with algorithms such as `thrust::transform`, `thrust::sort`, and `thrust::reduce` to implement common parallel operations such as map and histogram construction.

In `exercise.cu`, we have a familiar-looking C++ program which generates some random two-dimensional points, finds the centroid of those points, and then names which quadrant of the square centered about the centroid each point lies in.

We're interested in counting how many of these points are in each of the four quadrants.

The program looks like this:

    int main()
    {
      const size_t num_points = 10;
    
      std::vector<float2> points(num_points);
    
      generate_random_points(points);
    
      bounding_box box = compute_bounding_box(points);
      float2 center = box.center();
    
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
    }

If you've never seen `float2` before, don't worry. It's a simple built-in CUDA type that looks like this:

    struct float2
    {
      float x;
      float y;
    };

Just two `float`s.

Our task is to port this C++ program to run on the GPU using the [Thrust](thrust.github.com) library.

