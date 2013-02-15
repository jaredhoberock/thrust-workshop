Fun with Points!
================

Have you ever wondered how to build parallel programs that run on GPUs? It turns out it's easy when you have the [right tools](https://developer.nvidia.com/cuda-downloads).

In this post, we'll become familiar with algorithms such as `transform`, `sort`, and `reduce` to implement common parallel operations such as map and histogram construction. And they'll run on the GPU.

In `exercise.cu`, we have a C++ program which generates some random two-dimensional points, finds the centroid of those points, and then names which quadrant of the square centered about the centroid each point lies in.

We're dying to find out how many of these points are in each of the four quadrants.

At a high level, the program looks like this:

    int main()
    {
      const size_t num_points = 1 << 20; // one MILLION points!
    
      std::vector<float2> points(num_points);
    
      generate_random_points(points);
    
      float2 centroid = compute_centroid(points);
    
      std::vector<int> quadrants(points.size());
      classify(points, centroid, quadrants);
    
      std::vector<int> counts_per_quadrant(4);
      count_points_in_quadrants(points, quadrants, counts_per_quadrant);
    
      std::cout << "Per-quadrant counts:" << std::endl;
      std::cout << "  Bottom-left : " << counts_per_quadrant[0] << " points" << std::endl;
      std::cout << "  Bottom-right: " << counts_per_quadrant[1] << " points" << std::endl;
      std::cout << "  Top-left    : " << counts_per_quadrant[2] << " points" << std::endl;
      std::cout << "  Top-right   : " << counts_per_quadrant[3] << " points" << std::endl;
      std::cout << std::endl;
    }

Our task is to port this C++ program to run on the GPU using the [Thrust](thrust.github.com) algorithms library. Since the program is already broken down into a __high level description__ using functions with names like `generate_random_points` and `count_points_in_quadrants` that operate on __collections of data__, it'll be a breeze.

Plan of Attack
--------------

To port this program to run on the GPU, let's attack it in a series of steps. Since the program is already nicely factored into a series of function calls all we need to do is:

  1. Parallelize each call individually
  2. Point each parallel call at the GPU

Generating the Points
---------------------

Let's start out with taking a look at the inside of `generate_random_points`:

    void generate_random_points(std::vector<float2> &points)
    {
      // sequentially generate some random 2D points in the unit square
      for(int i = 0; i < points.size(); ++i)
      {
        float x = float(rand()) / RAND_MAX;
        float y = float(rand()) / RAND_MAX;
    
        points[i] = make_float2(x,y);
      }
    }

It's basically a sequential `for` loop that calls `rand` a bunch of times. Not very parallel.

To make matters worse, we know that the reason you get a different number each time you call `rand` is because there's some secret __implicit shared state__ inside that gets updated with each call. If we called `rand` a bunch of times in parallel from different threads, they might run into each other!

We'll need to __rethink our algorithm__ if we want to parallelize this operation.

Besides the stateful method used by `rand`, [it turns out](http://www.deshawresearch.com/resources_random123.html) another reasonable way to generate pseudorandom numbers is with a stateless integer hash, like this one right here:

    unsigned int hash(unsigned int x)
    {
      x = (x+0x7ed55d16) + (x<<12);
      x = (x^0xc761c23c) ^ (x>>19);
      x = (x+0x165667b1) + (x<<5);
      x = (x+0xd3a2646c) ^ (x<<9);
      x = (x+0xfd7046c5) + (x<<3);
      x = (x^0xb55a4f09) ^ (x>>16);
      return x;
    }

No state here -- to get a number, all we need to do is stick an integer in.

But what about that sequential `for` loop? That's where Thrust comes in. Thrust has a [large suite of algorithms](http://thrust.github.com/doc/group__algorithms.html) for solving parallel problems like this one.

In particular, we can use `thrust::tabulate` to call our `hash` function for each point in our set. Let's see how to hook it up:

    struct random_point
    {
      float2 operator()(unsigned int x)
      {
        return make_float2(float(hash(x)) / UINT_MAX, float(hash(2 * x)) / UINT_MAX);
      }
    };

    void generate_random_points(std::vector<float2> &points)
    {
      thrust::tabulate(points.begin(), points.end(), random_point());
    }

`thrust::tabulate` fills all the `points` from `begin` to `end` with a point created by `random_point`. Each time it calls the `random_point` [function object](http://en.wikipedia.org/wiki/Function_object#In_C_and_C.2B.2B), it passes the index of the element in question. The whole thing happens in parallel -- we have no idea in which order the points will be created. This gives Thrust a lot of flexibility in choosing how to execute the algorithm.

Finding the Centroid
--------------------

The next thing we need to do is take our points and find their centroid (average). The sequential code looks like this:

    float2 compute_centroid(const std::vector<float2> &points)
    {
      float2 sum = make_float2(0,0);
    
      // compute the mean
      for(int i = 0; i < points.size(); ++i)
      {
        sum = sum + points[i];
      }
    
      return make_float2(sum.x / points.size(), sum.y / points.size());
    }

Here, we're just summing up all the points and then dividing by the number at the end. Parallel programmers call this kind of operation a __reduction__ because we've taken a collection of things (points) and __reduced__ them to a single thing (the centroid).

With Thrust, reductions are easy -- we just call `reduce`:

    float2 compute_centroid(const std::vector<float2> &points)
    {
      float2 init = make_float2(0,0);
      float2 sum = thrust::reduce(points.begin(), points.end(), init);
      return make_float2(sum.x / points.size(), sum.y / points.size());
    }

We start by choosing an initial value for the reduction -- `init` -- which initializes our sum to zero. Then we tell Thrust we want to `reduce` all the points from `begin` to `end` using `init` as the initial value of the sum. (If `points` was easy, `reduce` would simply return `init`.)

By default, `reduce` assumes we want to compute a mathematical sum, but it's actually a __higher order function__ like `tabulate`. If we wanted to compute some other kind of reduction besides a sum, we could pass an [associative](http://en.wikipedia.org/wiki/Associativity) function object that told Thrust how to reduce the points.

