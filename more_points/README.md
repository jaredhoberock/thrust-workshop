# TL;DR

XXX we can even build spatial data structures like trees in parallel!

# More Points

XXX let's build upon the [fun_with_points](../fun_with_points) example.

In this post, we'll become familiar with algorithms such as `exclusive_scan`
and `lower_bound` to build sophisticated data structures in parallel. As
always, we'll structure the code such that it can run anywhere we have parallel
resources.

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
      const size_t num_points = 10000000;
      size_t num_tree_levels = XXX;

      std::vector<float2> points(num_points);

      generate_random_points(points);

      bbox bounds = compute_bounding_box(points);

      std::vector<int> tags(num_points);

      compute_tags(points, bounds, tags);

      sort_points_by_tag(points, tags);

      std::vector<int> nodes;
      std::vector<int2> leaves;
      build_tree(tags, bounds, num_tree_levels, nodes, leaves);
    }

Our tree data structure is just an array of (interior) nodes and a list of (terminal) leaves.

The `build_tree` function is itself composed of several steps, so let's take a look inside:

    void build_tree(const std::vector<int> &tags,
                    const bbox &bounds,
                    size_t num_tree_levels,
                    std::vector<int> &nodes,
                    std::vector<int2> &leaves)
    {
      std::vector<int> active_nodes(1,0);
      
      // build the tree one level at a time, starting at the root
      for(int level = 1; !active_nodes.empty() && level < num_tree_levels; ++level)
      {
        // each node has four children since this is a quad tree
        std::vector<int> children(4 * active_nodes.size());

        compute_child_tag_masks(active_nodes, level, num_tree_levels, children);

        std::vector<int> lower_bounds(children.size());
        std::vector<int> upper_bounds(children.size());
        compute_child_bounds(tags, children, level, num_tree_levels, lower_bounds, upper_bounds);

        // mark each child as either empty, an interior node, or a leaf
        std::vector<int> child_node_kind(children.size(), 0);
        classify_children(children, lower_bounds, upper_bounds, child_node_kind);

        std::vector<int> nodes_on_this_level(child_node_kind.size());
        std::vector<int> leaves_on_this_level(child_node_kind.size());

        enumerate_nodes_and_leaves(child_node_kind, nodes_on_this_level, leaves_on_this_level);

        create_child_nodes(child_node_kind, nodes_on_this_level, leaves_on_this_level, nodes);

        create_leaves(child_node_kind, leaves_on_this_level, lower_bounds, upper_bounds, leaves);

        activate_nodes_for_next_level(child_node_kind, children, active_nodes);
      }
    }

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

In order to compute a single box which is large enough to contain all of our points, we need to inspect them all and *reduce* them into a single value -- the box.

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

We just loop through the points and make sure the box is large enough to contain each one. If the box isn't large enough along a particular dimension, we extend it such that it just contains the point.

At first glance, it may seem difficult to parallelize this operation because each iteration incrementally builds off of the last one. In fact, it's possible to cast this operation as a *reduction*.

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

