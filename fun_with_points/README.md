Fun with Points!
================

Have you ever wondered how to build parallel programs that run on GPUs? It turns out it's easy when you have the [right tools](https://developer.nvidia.com/cuda-downloads).

In this post, we'll become familiar with algorithms such as `transform`, `sort`, and `reduce` to implement common parallel operations such as map and histogram construction. And they'll run on the GPU.

In `exercise.cu`, we have a familiar-looking C++ program which generates some random two-dimensional points, finds the centroid of those points, and then names which quadrant of the square centered about the centroid each point lies in.

We're dying to find out how many of these points are in each of the four quadrants.

At a high level, the program looks like this:

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

Our task is to port this C++ program to run on the GPU using the [Thrust](thrust.github.com) library. Since the program is already broken down into a __high level description__ using functions with names like `generate_random_points` and `count_points_in_quadrants` that operate on __collections of data__, it'll be a breeze.

