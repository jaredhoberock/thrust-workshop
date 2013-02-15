TL;DR
=====

    $ git clone git://github.com/jaredhoberock/thrust-workshop
    $ cd thrust-workshop/fun_with_points
    $ scons
    $ ./exercise
    $ ./spoilers

Fun with Points!
================

Have you ever wondered how to build parallel programs that run on GPUs? It turns out it's easy when you have the [right tools](https://developer.nvidia.com/cuda-downloads).

In this post, we'll become familiar with algorithms such as `transform`, `sort`, and `reduce` to implement common parallel operations such as map and histogram construction. And they'll run on the GPU.

In `exercise.cu`, we have a C++ program which generates some random two-dimensional points, finds the centroid of those points, and then names which quadrant of the square centered about the centroid each point lies in.

We're dying to find out how many of these points are in each of the four quadrants.

At a high level, the program looks like this:

    int main()
    {
      const size_t num_points = 1000000;
    
      std::vector<float2> points(num_points);
    
      generate_random_points(points);
    
      float2 centroid = compute_centroid(points);
    
      std::vector<int> quadrants(points.size());
      classify_points_by_quadrant(points, centroid, quadrants);
    
      std::vector<int> counts_per_quadrant(4);
      count_points_in_quadrants(points, quadrants, counts_per_quadrant);
    
      std::cout << "Per-quadrant counts:" << std::endl;
      std::cout << "  Bottom-left : " << counts_per_quadrant[0] << " points" << std::endl;
      std::cout << "  Bottom-right: " << counts_per_quadrant[1] << " points" << std::endl;
      std::cout << "  Top-left    : " << counts_per_quadrant[2] << " points" << std::endl;
      std::cout << "  Top-right   : " << counts_per_quadrant[3] << " points" << std::endl;
      std::cout << std::endl;
    }

Let's port this C++ program to run on the GPU using the [Thrust](thrust.github.com) algorithms library. Since the program is already broken down into a __high level description__ using functions with names like `generate_random_points` and `count_points_in_quadrants` that operate on __collections of data__, it'll be a breeze.

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

We start by choosing an initial value for the reduction -- `init` -- which initializes our sum to zero. Then we tell Thrust we want to `reduce` all the points from `begin` to `end` using `init` as the initial value of the sum. (If `points` was empty, `reduce` would simply return `init`.)

By default, `reduce` assumes we want to compute a mathematical sum, but it's actually a __higher order function__ like `tabulate`. If we wanted to compute some other kind of reduction besides a sum, we could pass an [associative](http://en.wikipedia.org/wiki/Associativity) function object that defines what it means to combine two points together.

Classifying each Point
----------------------

Before we can count how many points are in each of the four quadrants, we need to figure out which quadrant each point is in. 

An easy way to do this is to compare each point to the `centroid`. The sequential code looks like this:

    void classify_points_by_quadrant(const std::vector<float2> &points, float2 centroid, std::vector<int> &quadrants)
    {
      // classify each point relative to the centroid
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

We compare each point's `x` and `y` coordinate to the `centroid`, and compute a number between `0` and `3` using some fancy bitwise manipulation with the `|` operation.

In this example, the important thing to realize is that unlike our sequential `for` loop from the last example, none of the iterations of this `for` loop have any __dependency__ on any other iteration.

Sometimes we these kinds of operations [__embarassingly parallel__](http://en.wikipedia.org/wiki/Embarassingly_parallel), because parallelizing them is embarassingly easy. Another term for this operation is a __parallel map__ because each thing (point) in our collection gets __mapped__ to another thing (a number).

With Thrust, we can compute parallel map operations using `transform` (`map` means [`something else`](http://en.wikipedia.org/wiki/Std::map) in C++):

    struct classify_point
    {
      float2 center;
    
      classify_point(float2 c)
      {
        center = c;
      }
    
      unsigned int operator()(float2 p)
      {
        return (p.x <= center.x ? 0 : 1) | (p.y <= center.y ? 0 : 2);
      }
    };
    
    void classify_points_by_quadrant(const std::vector<float2> &points, float2 center, std::vector<int> &quadrants)
    {
      thrust::transform(points.begin(), points.end(), quadrants.begin(), classify_point(center));
    }

`transform` works kind of like `tabulate`, but instead of automatically generating a series of integer indices for us, `transform` passes each point from `begin` to `end` to our `classify_point` function object.

Each call to `classify_point` performs the same operation as each iteration of our sequential `for` loop. However, instead of assigning the result to `quadrants[i]`, `classify_point` returns it. Since we passed `quadrants.begin()` to `transform`, it knows where each result should go.

