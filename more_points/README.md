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

# 

