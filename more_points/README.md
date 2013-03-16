# TL;DR

XXX we can even build spatial data structures like trees in parallel!

# More Points

Now that we've had our [fun with points](../fun_with_points), it's time to get
serious. Let's use Thrust to get some work done by organizing our points into a
spatial data structure. In this post, we'll become familiar with algorithms
such as `exclusive_scan` and `lower_bound` to build sophisticated data
structures in parallel. Just like before, we'll structure the code such that it
can run anywhere we have parallel resources.

In [exercise.cu](exercise.cu), we have a C++ program that generates some random
two-dimensional points, finds the bounding box of those points, and then
iteratively subdivides that box to build a [hierarchical tree structure](http://en.wikipedia.org/wiki/Quadtree). We
could use this kind of data structure later for performing spatial queries like
[detecting collisions](http://en.wikipedia.org/wiki/Collision_detection), or
[finding nearby neighbors](http://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm),
but for now we'll keep it simple and concentrate on just building the thing.

At a high level, the program looks like this:

    int main()
    {
      const int num_points = 10000000;
      int max_level = XXX;
      int threshold = YYY;

      std::vector<float2> points(num_points);
      generate_random_points(points);

      bbox bounds = compute_bounding_box(points);

      std::vector<int> tags(num_points);
      compute_tags(points, bounds, max_level, tags);

      std::vector<int> indices(num_points);
      sort_points_by_tag(tags, indices);

      std::vector<int> nodes;
      std::vector<int2> leaves;
      build_tree(tags, bounds, max_level, threshold, nodes, leaves);
    }

We'll describe what's going on with the `tags` later.

Our tree data structure is just an array of nodes and a list of leaves. Each leaf indexes a contiguous piece of the `indices` array.

The implementation of `build_tree` function is fairly complex. We'll peek inside later.

# Massaging the Data

Before we can dive into building our tree, we first need to generate our input and massage the data into a format which makes tree construction easy.

## Generating the Points

First, we'll generate some random 2D points in parallel just like we did in the [`fun_with_points`](../fun_with_points) example:

    std::vector<float2> points(num_points);

    generate_random_points(points);

We'll keep our `points` data on the CPU in a `std::vector` for now. Later, we'll move it to the GPU with a `thrust::device_vector`. Inside of `generate_random_points` is just a call to `thrust::tabulate` which produces random points.

## Bounding the Points

The next thing we do is compute a ["bounding box"](http://en.wikipedia.org/wiki/Minimum_bounding_box) for our points. If you're unfamiliar with the idea, a bounding box is a box which contains or "bounds" all of our points. You can think of it as describing the geometric boundaries of our problem set.

For our purposes, a `bbox` is just two points which specify the coordinates of the extremal corners of the box along the two coordinate axes x and y:

    struct bbox
    {
      float xmin, xmax;
      float ymin, ymax;
    
      // initialize empty box
      inline __host__ __device__
      bbox() : xmin(FLT_MAX), xmax(-FLT_MAX), ymin(FLT_MAX), ymax(-FLT_MAX)
      {}
      
      // initialize a box containing a single point
      inline __host__ __device__
      bbox(const float2 &p) : xmin(p.x), xmax(p.x), ymin(p.y), ymax(p.y)
      {}
    };

It's defined in [`util.h`](util.h) in the source.

In order to compute a single box which is large enough to contain all of our points, we need to inspect them all and __reduce__ them into a single value -- the box.

Here's what the sequential code looks like:

    // start with an empty box
    bbox bounds;

    // incrementally enlarge the box to include each point
    for(int i = 0; i < num_points; ++i)
    {
      float2 p = points[i];
      if(p.x < bounds.xmin) bounds.xmin = p.x;
      if(p.x > bounds.xmax) bounds.xmax = p.x;
      if(p.y < bounds.ymin) bounds.ymin = p.y;
      if(p.y > bounds.ymax) bounds.ymax = p.y;
    }

We just loop through the points and make sure the box is large enough to contain each one. If the box isn't large enough along a particular dimension, we extend it such that it is just large enough to contain the point.

At first glance, it may seem difficult to parallelize this operation because each iteration incrementally builds off of the last one. In fact, it's possible to cast this operation as a __reduction__.

In the [`fun_with_points`](../fun_with_points) example, we used `thrust::reduce` to compute the average of a collection of points. Here, the result was the same type as the input -- the average of a collection of points is still a point.

In this case, we'd like to compute a result (a `bbox`) which is a different type than the input (a collection of `float2`s). That's okay -- as long as the type of the input is convertible to the result (note the second constructor of `bbox`) `thrust::reduce` will make it work.

To implement this bounding box reduction, we'll introduce a functor which merges two `bbox`s together as the fundamental reduction step. The resulting `bbox` is large enough to hold the two inputs:

    struct merge_boxes
    {
      inline __host__ __device__
      bbox operator()(const bbox &a, const bbox &b) const
      {
        bbox result;
        result.xmin = min(a.xmin, b.xmin);
        result.xmax = max(a.xmax, b.xmax);
        result.ymin = min(a.ymin, b.ymin);
        result.ymax = max(a.ymax, b.ymax);
        return result;
      }
    };

Internally, the way the reduction will work is to create, for each point, a single `bbox` which will bound only that point. Then, the reduction will merge `bbox`s in pairs until finally a single one which bounds everything results:

    bbox compute_bounding_box(const std::vector<float2> &points)
    {
      // we pass an empty bounding box for thrust::reduce's init parameter
      bbox empty;
      return thrust::reduce(points.begin(), points.end(), empty, merge_bboxes());
    }

## Linearizing the Points

The next step in preparing our data for tree construction is to augment its representation. The basic idea behind the tree construction process is to pose it as a spatial sorting problem. But sorts are one dimensional and we have two dimensional data. What does it mean to sort 2D points?

Since we want to organize our points spatially, we'd like a sorting solution which preserves spatial locality. In other words, we want points that are nearby in 2D to be near each other in the one dimensional `points` array.

This is actually a whole lot easier than it sounds. The basic idea is to "tag" each point with its [index](http://en.wikipedia.org/wiki/Morton_code) along a [space-filling curve](http://en.wikipedia.org/wiki/Space_filling_curve).

It turns out that points with nearby tags are also nearby in 2D! That means if we sort our collection of `points` by their `tags`, we'll order them in a way that encourages spatial locality which will be important to the tree building process later.

The sequential code is pretty simple. It just associates with each point a tag:

    std::vector<int> tags(points.size());

and computes them with the `compute_tags` function:

    void compute_tags(const std::vector<float2> &points,
                      const bbox &bounds,
                      int max_level,
                      std::vector<int> &tags)
    {
      for(int i = 0; i < points.size(); ++i)
      {
        float2 p = points[i];
        tags[i] - point_to_tag(p, bounds, max_level);
      }
    }

The `point_to_tag` computation takes a point `p`, the `bounds` of the entire collection of `points`, and the index of the tree's `max_level` and computes the point's spatial code. If you're interested in the details, you can peek inside [`util.h`](util.h) where it's defined.

This operation looks a lot like the point classification problem from the [`fun_with_points`](../fun_with_points) exercise. We know that we can parallelize embarrassingly parallel operations like these with `thrust::transform`:

    struct classify_point
    {
      bbox box;
      int max_level;

      classify_point(const bbox &bounds, int max_level) :
        box(bounds),
        max_level(max_level)
      {}

      inline __device__ __host__
      int operator()(const float2 &p)
      {
        return point_to_tag(p, box, max_level);
      }
    };

    void compute_tags(const std::vector<float2> &points,
                      const bbox &bounds,
                      std::vector<int> &tags)
    {
      thrust::transform(points.begin(), points.end(),
                        tags.begin(),
                        classify_point(bounds, max_level));
    }

The only thing we need to do is introduce the `classify_point` functor whose job it is to call the `point_to_tag` function.

## Sorting by Tag

Now that each point has a spatial `tag`, we can organize them spatially just by sorting the `points` by their `tags`.

The sequential CPU code does it this way:

    struct compare_tags
    {
      template <typename Pair>
      inline bool operator()(const Pair &p0, const Pair &p1) const
      {
        return p0.first < p1.first;
      }
    };

    void sort_points_by_tag(std::vector<int> &tags, std::vector<float2> &points)
    {
      // introduce a temporary array of pairs for sorting purposes
      std::vector<std::pair<int,int> > tag_index_pairs(num_points);
      for(int i = 0; i < num_points; ++i)
      {
        tag_index_pairs[i].first = tags[i];
        tag_index_pairs[i].second = i;
      }

      std::sort(tag_index_pairs.begin(), tag_index_pairs.end(), compare_tags());

      // copy sorted data back into input arrays
      for(int i = 0; i < num_points; ++i)
      {
        tags[i]    = tag_index_pairs[i].first;
        indices[i] = tag_index_pairs[i].second;
      }
    }

Which is a pretty roundabout way of coaxing a key-value sort out of `std::sort`. With Thrust we can do it in parallel with just a call to `thrust::sort_by_key`:

    void sort_points_by_tag(std::vector<int> &tags, std::vector<float2> &points)
    {
      thrust::sort_by_key(tags.begin(), tags.end(), points.begin());
    }

# Building the Tree

Now that we've got our points nice and organized, it's time to build the tree! We'll build each level of the tree one by one, and building each level requires a series of steps.

Here's the high-level overview of the process:

    void build_tree(const std::vector<int> &tags,
                    const bbox &bounds,
                    int max_level,
                    int threshold,
                    std::vector<int> &nodes,
                    std::vector<int2> &leaves)
    {
      std::vector<int> active_nodes(1,0);
      
      // build the tree one level at a time, starting at the root
      for(int level = 1; !active_nodes.empty() && level <= max_level; ++level)
      {
        // each node has four children since this is a quad tree
        std::vector<int> children(4 * active_nodes.size());

        compute_child_tag_masks(active_nodes, level, max_level, children);

        std::vector<int> lower_bounds(children.size());
        std::vector<int> upper_bounds(children.size());
        find_child_bounds(tags, children, level, max_level, lower_bounds, upper_bounds);

        // mark each child as either empty, an interior node, or a leaf
        std::vector<int> child_node_kind(children.size(), 0);
        classify_children(children, lower_bounds, upper_bounds, level, max_level, threshold, child_node_kind);

        // enumerate the nodes and leaves at this level
        std::vector<int> nodes_on_this_level(child_node_kind.size());
        std::vector<int> leaves_on_this_level(child_node_kind.size());

        std::pair<int,int> num_nodes_and_leaves_on_this_level =
          enumerate_nodes_and_leaves(child_node_kind, nodes_on_this_level, leaves_on_this_level);

        create_child_nodes(child_node_kind, nodes_on_this_level, leaves_on_this_level, leaves.size(), nodes);

        create_leaves(child_node_kind, leaves_on_this_level, lower_bounds, upper_bounds, num_nodes_and_leaves_on_this_level.second, leaves);

        activate_nodes_for_next_level(children, child_node_kind, active_nodes);

        activate_nodes_for_next_level(children, child_node_kind, num_nodes_and_leaves_on_this_level.first, active_nodes);
      }
    }

You can see that it takes as input the information we computed in the prior
steps (`tags`, `bounds`) and some tweakable knobs (`max_level`, `threshold`)
and produces two arrays: `nodes` and `leaves`.

Each element of the `nodes` array is an index which identifies whether the node is empty, or whether it
refers to a terminal leaf, or an interior node. When it is a leaf, the index encodes where in the `leaves` array it lives.
When it is a node, 

