TL;DR
=====

Building __parallel programs__ is easy with [Thrust's](http://thrust.github.com) __power tools__ like parallel __maps__, __sorts__, and __reductions__.

    $ git clone git://github.com/jaredhoberock/thrust-workshop
    $ cd thrust-workshop/fun_with_points
    $ scons
    $ ./exercise
    $ ./spoilers

(Requires [scons](http://www.scons.org/) and [CUDA](https://developer.nvidia.com/cuda-downloads).)

Fun with Points!
================

Have you ever wondered how to build programs that run on parallel processors like GPUs? It turns out it's easy if you __think parallel__.

In this post, we'll become familiar with algorithms such as `transform`, `sort`, and `reduce` to implement common parallel operations such as map and histogram construction. And they'll run on the GPU.

In [`exercise.cu`](exercise.cu), we have a C++ program which generates some random two-dimensional points, finds the centroid of those points, and then names which quadrant of the square centered about the centroid each point lies in.

We're dying to find out how many of these points are in each of the four quadrants.

At a high level, the program looks like this:

    int main()
    {
      const size_t num_points = 10000000;
    
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

Let's port this C++ program to run on the GPU using the [Thrust](http://thrust.github.com) algorithms library. Since the program is already broken down into a __high level description__ using functions with names like `generate_random_points` and `count_points_in_quadrants` that operate on __collections of data__, it'll be a breeze.

To follow along with this post in [`example.cu`](example.cu), type `scons example` into your command line to build the program, and `./example` to run it. You'll need to install [CUDA](https://developer.nvidia.com/cuda-downloads) to get NVIDIA's compiler to compile `.cu` files and [SCons](http://www.scons.org/), which is the build system we'll use.

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

It's basically a __sequential__ `for` loop that calls `rand` a bunch of times. That means that each iteration of this loop gets executed one at a time, in order. Not very parallel.

To make matters worse, we know that the reason you get a different number each time you call `rand` is because there's some secret __implicit shared state__ inside that gets updated with each call. If we called `rand` a bunch of times in parallel all at once, they might run into each other!

We'll need to __rethink our algorithm__ if we want to parallelize this operation.

Besides the stateful method used by `rand`, [it turns out](http://www.deshawresearch.com/resources_random123.html) another reasonable way to generate pseudorandom numbers is with a stateless integer hash, like this one right here:

    __host__ __device__
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

No state here -- to get a number, all we need to do is stick an integer in. But what's this `__host__ __device__` business? That's there to let the CUDA C++ compiler know that the `hash` function can be called from either the `__host__` (the CPU) or the `__device__` (the GPU). Without it, our program won't compile.

But what about that sequential `for` loop? That's where Thrust comes in. Thrust has a [large suite of algorithms](http://thrust.github.com/doc/group__algorithms.html) for solving parallel problems like this one.

In particular, we can use `thrust::tabulate` to call our `hash` function for each point in our set. Let's see how to hook it up:

    struct random_point
    {
      __host__ __device__
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

Sometimes we call these kinds of operations [__embarassingly parallel__](http://en.wikipedia.org/wiki/Embarassingly_parallel), because parallelizing them is embarassingly easy. Another term for this operation is a __parallel map__ because each thing (point) in our collection gets __mapped__ to another thing (a number).

With Thrust, we can compute parallel map operations using `transform` (`map` means [`something else`](http://en.wikipedia.org/wiki/Std::map) in C++):

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
    
    void classify_points_by_quadrant(const std::vector<float2> &points, float2 center, std::vector<int> &quadrants)
    {
      thrust::transform(points.begin(), points.end(), quadrants.begin(), classify_point(center));
    }

`transform` works kind of like `tabulate`, but instead of automatically generating a series of integer indices for us, `transform` passes each point from `begin` to `end` to our `classify_point` function object.

Each call to `classify_point` performs the same operation as each iteration of our sequential `for` loop. However, instead of assigning the result to `quadrants[i]`, `classify_point` returns it. Since we passed `quadrants.begin()` to `transform`, it knows where each result should go.

Tallying Up Each Quadrant
-------------------------

We're almost done. The last thing we need to do is answer is our original burning question of how many of our points are in each quadrant.

When we have a set of buckets (quadrants) and we want to count how many things (points) fall in each, we sometimes say we want to compute a __histogram__. 

The sequential code counts up the points by looping over them and incrementing a counter in each bucket:

    void count_points_in_quadrants(std::vector<float2> &points, std::vector<int> &quadrants,
                                   std::vector<int> &counts_per_quadrant)
    {
      // sequentially compute a histogram
      for(int i = 0; i < quadrants.size(); ++i)
      {
        int q = quadrants[i];
    
        // increment the number of points in this quadrant
        counts_per_quadrant[q]++;
      }
    }

Hmm... there's that __shared state__ problem again. The iterations of this loop __depend__ on each other, because we expect many points to fall into the same quadrant and __contend__ to increment the same counter. If we tried to perform all these increments at once in parallel with an algorithm like `transform`, they would all run into each other! We'll have to figure out a different way to build this histogram.

You may notice that our `count_points_in_quadrants` function takes `points` as a parameter, but it doesn't do anything with them. That's a hint that to parallelize this operation, we'll need to think about __reorganizing__ our input data to make our job easier.

This problem of counting up a collection of items might remind you of our earlier use of `reduce`. Like before, we have a collection of things (points in a quadrant) and we want to reduce them to a single thing (the number of points in each quadrant).

But it doesn't make sense to just call `reduce` again -- it seems like `reduce` could only count up the number of points in a single quadrant. So we'd have to call `reduce` several times -- once for each quadrant. But how do we find the points associated with a particular quadrant and pick them out for `reduce`?

It turns out if we're willing to __sort__ our data, we can bring all the points from a particular quadrant together so that we can __reduce__ them all at once. We call this kind of operation __sorting by key__ because we're sorting a collection of things (points) by a key (quadrant number) associated with each of them.

This is super easy with the `sort_by_key` algorithm:

    thrust::sort_by_key(quadrants.begin(), quadrants.end(), points.begin());

Here, we're telling Thrust to consider all the quadrant numbers from `begin` to `end` as sorting keys for the points beginning at `points.begin()`. Afterwards, both collections will be sorted according to the keys.

So now our points are sorted, but so what? How does sorting help us count them? How do we use `reduce`?

Since by sorting we've brought all of the things (points) with the same key (quadrant number) together, we can do a special kind of __reducing by key__ operation to count them all at once:

    thrust::reduce_by_key(quadrants.begin(), quadrants.end(),
                          thrust::constant_iterator<int>(1),
                          thrust::discard_iterator<>(),
                          counts_per_quadrant.begin());

Whoa... what just happened?

Let's break it down piece by piece:

Like `sort_by_key`, `reduce_by_key` takes a collection of __keys__:

    quadrants.begin(), quadrants.end() // The key is the quadrant number again.

And a collection of __values__:

    thrust::constant_iterator<int>(1) // The endlessly repeating sequence 1, 1, 1, ...

And __reduces__ each span of contiguous values with the same key. For each span, it returns the key:

    thrust::discard_iterator<>() // We're not interested in retaining the key; just drop it on the floor.

And the reduced value:

    counts_per_quadrant.begin() // The result we're actually interested in.

Just like `reduce`, `reduce_by_key`'s default reduction is a sum. So for each key, we're summing up the value `1`, which comes from that `constant_iterator` thing.

[Iterators](http://en.wikipedia.org/wiki/Iterator) are like pointers. They're how Thrust knows where to find the inputs and outputs to each algorithm. The `.begin()` and `.end()` thingies we've used are examples but you can also [get fancy](http://thrust.github.com/doc/group__fancyiterator.html) with iterators like `constant_iterator` and `discard_iterator` to generate data on the fly.

Here's the whole function:

    void count_points_in_quadrants(std::vector<float2> &points, std::vector<int> &quadrants,
                                   std::vector<int> &counts_per_quadrant)
    {
      // sort points by quadrant
      thrust::sort_by_key(quadrants.begin(), quadrants.end(), points.begin());
    
      // count points in each quadrant
      thrust::reduce_by_key(quadrants.begin(), quadrants.end(),
                            thrust::constant_iterator<int>(1),
                            thrust::discard_iterator<>(),
                            counts_per_quadrant.begin());
    }

Pointing it at the GPU
----------------------

So we're all done, right? Not quite. Remember I said that we'd attack our porting problem in two parts: first by reorganizing our code into high-level parallel operations, and then by pointing those parallel operations at the GPU.

Even though we've rewritten our program to use parallel Thrust algorithms, we're still not done yet. By default, whenever the inputs to Thrust algorithms come from things like `std::vector`, Thrust executes those algorithms sequentially on the CPU.

Fortunately, this is the easiest part. To point Thrust at the GPU, all we need to do is `s/std::vector/thrust::device_vector/` and we're set. `device_vector` is a special kind of vector container that sticks its data in memory that's easy for the GPU to access. Whenever a Thrust algorithm gets its input and output from a `device_vector`, that algorithm will *execute on the GPU in parallel*.

