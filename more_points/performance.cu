#include <thrust/device_vector.h>
#include <thrust/tabulate.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <iostream>
#include <iomanip>
#include <bitset>
#include <algorithm>
#include <cfloat>
#include "time_invocation_cuda.hpp"
#include "util.h"

// Markers
enum { NODE = 1, LEAF = 2, EMPTY = 4 };

// Utility functions to encode leaves and children in single int
// are defined in util.h:
//   bool is_empty(int id);
//   bool is_node(int id);
//   bool is_leaf(int id);
//   int get_empty_id();
//   int get_leaf_id(int offset);
//   int get_leaf_offset(int id);

// Operator which merges two bounding boxes.
struct merge_bboxes
{
  inline __host__ __device__
  bbox operator()(const bbox &b0, const bbox &b1) const
  {
    bbox bounds;
    bounds.xmin = min(b0.xmin, b1.xmin);
    bounds.xmax = max(b0.xmax, b1.xmax);
    bounds.ymin = min(b0.ymin, b1.ymin);
    bounds.ymax = max(b0.ymax, b1.ymax);
    return bounds;
  }
};

bbox compute_bounding_box(const thrust::device_vector<float2> &points)
{
  return thrust::reduce(points.begin(), points.end(), bbox(), merge_bboxes());
}


// Classify a point with respect to the bounding box.
struct classify_point
{
  bbox box;
  int max_level;

  // Create the classifier
  classify_point(const bbox &b, int lvl) : box(b), max_level(lvl) {}

  // Classify a point
  inline __device__ __host__
  int operator()(const float2 &p) { return point_to_tag(p, box, max_level); }
};

void compute_tags(const thrust::device_vector<float2> &points, const bbox &bounds, int max_level, thrust::device_vector<int> &tags)
{
  thrust::transform(points.begin(), 
                    points.end(), 
                    tags.begin(), 
                    classify_point(bounds, max_level));
}


void sort_points_by_tag(thrust::device_vector<int> &tags, thrust::device_vector<int> &indices)
{
  thrust::sequence(indices.begin(), indices.end());
  thrust::sort_by_key(tags.begin(), tags.end(), indices.begin());
}


struct expand_active_nodes
{
  int level, max_level;
  const int *nodes;
  
  expand_active_nodes(int lvl, int max_lvl, const int *nodes) : level(lvl), max_level(max_lvl), nodes(nodes) {}
  
  inline __device__ __host__
  int operator()(int idx) const
  {
    int tag = nodes[idx/4];
    int mask = (idx&3) << (2*(max_level-level));
    return tag | mask;
  }
};

struct add
{
  typedef int result_type;
  int val;
  add(int v) : val(v) {}
  inline __device__ __host__ int operator()(int x) const { return x+val-1; }
};

struct mark_nodes
{
  int threshold;
  int last_level;
  
  mark_nodes(int threshold, int last_level) : threshold(threshold), last_level(last_level) {}

  template <typename tuple_type>
  inline __device__ __host__
  int operator()(const tuple_type &t) const
  {
    int lower_bound = thrust::get<0>(t);
    int upper_bound = thrust::get<1>(t);
    
    int count = upper_bound - lower_bound;
    if (count == 0)
    {
      return EMPTY;
    }
    else if (last_level || count < threshold)
    {
      return LEAF;
    }
    else
    {
      return NODE;
    }
  }
};

struct write_nodes
{
  int num_nodes, num_leaves;

  write_nodes(int num_nodes, int num_leaves) : 
    num_nodes(num_nodes), num_leaves(num_leaves) 
  {}

  template <typename tuple_type>
  inline __device__ __host__
  int operator()(const tuple_type &t) const
  {
    int node_type = thrust::get<0>(t);
    int node_idx  = thrust::get<1>(t);
    int leaf_idx  = thrust::get<2>(t);

    if (node_type == EMPTY)
    {
      return get_empty_id();
    }
    else if (node_type == LEAF)
    {
      return get_leaf_id(num_leaves + leaf_idx);
    }
    else
    {
      return num_nodes + 4 * node_idx;
    }
  }
};

struct make_leaf
{
  typedef int2 result_type;
  template <typename tuple_type>
  inline __device__ __host__
  int2 operator()(const tuple_type &t) const
  {
    int x = thrust::get<0>(t);
    int y = thrust::get<1>(t);

    return make_int2(x, y);
  }
};

void build_tree(const thrust::device_vector<int> &tags,
                const bbox &bounds,
                size_t max_level,
                int threshold,
                thrust::device_vector<int> &nodes,
                thrust::device_vector<int2> &leaves)
{
  thrust::device_vector<int> active_nodes(1,0);

  int num_nodes = 0, num_leaves = 0;

  // Build the tree one level at a time, starting at the root
  for (int level = 1 ; !active_nodes.empty() && level <= max_level ; ++level)
  {
    // Number of nodes to process at this level
    int num_active_nodes = static_cast<int>(active_nodes.size());

    /******************************************
     * 6. Calculate children                  *
     ******************************************/

    // New children: 4 quadrants per active node = 4 children
    thrust::device_vector<int> children(4*num_active_nodes);

    // For each active node, generate the tag mask for each of its 4 children
    thrust::transform(thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(4*num_active_nodes),
                      children.begin(),
                      expand_active_nodes(level, max_level, thrust::raw_pointer_cast(&active_nodes.front())));

    /******************************************
     * 7. Determine interval for each child   *
     ******************************************/

    // For each child we need interval bounds
    thrust::device_vector<int> lower_bounds(children.size());
    thrust::device_vector<int> upper_bounds(children.size());

    // Locate lower and upper bounds for points in each quadrant
    thrust::lower_bound(tags.begin(),
                        tags.end(),
                        children.begin(),
                        children.end(),
                        lower_bounds.begin());

    add add_step(1 << 2*(max_level-level));
    thrust::upper_bound(tags.begin(),
                        tags.end(),
                        thrust::make_transform_iterator(children.begin(), add_step),
                        thrust::make_transform_iterator(children.end(), add_step),
                        upper_bounds.begin());

    /******************************************
     * 8. Mark each child as empty/leaf/node  *
     ******************************************/

    // Mark each child as either empty, a node, or a leaf
    thrust::device_vector<int> markers(children.size(), 0);

    thrust::transform(thrust::make_zip_iterator(
                          thrust::make_tuple(lower_bounds.begin(), upper_bounds.begin())),
                      thrust::make_zip_iterator(
                          thrust::make_tuple(lower_bounds.end(), upper_bounds.end())),
                      markers.begin(),
                      mark_nodes(threshold, level == max_level));

    /******************************************
     * 9. Enumerate nodes and leaves          *
     ******************************************/

    // Enumerate the nodes and leaves at this level
    thrust::device_vector<int> level_nodes(markers.size());
    thrust::device_vector<int> level_leaves(markers.size());

    // Enumerate nodes at this level
    thrust::transform_exclusive_scan(markers.begin(), 
                                     markers.end(), 
                                     level_nodes.begin(), 
                                     is_a<NODE>(), 
                                     0, 
                                     thrust::plus<int>());
    int num_level_nodes = level_nodes.back() + (markers.back() == NODE ? 1 : 0);

    // Enumerate leaves at this level
    thrust::transform_exclusive_scan(markers.begin(), 
                                     markers.end(), 
                                     level_leaves.begin(), 
                                     is_a<LEAF>(), 
                                     0, 
                                     thrust::plus<int>());
    int num_level_leaves = level_leaves.back() + (markers.back() == LEAF ? 1 : 0);

    /******************************************
     * 10. Add the children to the node list  *
     ******************************************/

    // Add these children to the list of nodes
    nodes.resize(num_nodes + children.size());

    thrust::transform(thrust::make_zip_iterator(
                          thrust::make_tuple(
                              markers.begin(), level_nodes.begin(), level_leaves.begin())),
                      thrust::make_zip_iterator(
                          thrust::make_tuple(
                              markers.end(), level_nodes.end(), level_leaves.end())),
                      nodes.begin() + num_nodes,
                      write_nodes(num_nodes + 4 * num_active_nodes, num_leaves));

    // Update the number of nodes
    num_nodes += 4 * num_active_nodes;

    /******************************************
     * 11. Add the leaves to the leaf list    *
     ******************************************/

    // Add child leaves to the list of leaves
    leaves.resize(num_leaves + num_level_leaves);
    thrust::scatter_if(thrust::make_transform_iterator(
                           thrust::make_zip_iterator(
                               thrust::make_tuple(lower_bounds.begin(), upper_bounds.begin())),
                           make_leaf()),
                       thrust::make_transform_iterator(
                           thrust::make_zip_iterator(
                               thrust::make_tuple(lower_bounds.end(), upper_bounds.end())),
                           make_leaf()),
                       level_leaves.begin(),
                       markers.begin(),
                       leaves.begin() + num_leaves,
                       is_a<LEAF>());

    // Update the number of leaves
    num_leaves += num_level_leaves;

    /******************************************
     * 12. Set the nodes for the next level   *
     ******************************************/
    
    // Set active nodes for the next level to be all the childs nodes from this level
    active_nodes.resize(num_level_nodes);

    thrust::copy_if(children.begin(),
                    children.end(),
                    markers.begin(),
                    active_nodes.begin(),
                    is_a<NODE>());

    // Update the number of active nodes.
    num_active_nodes = num_level_nodes;
  }
}

void run_experiment(thrust::device_vector<float2> *points,
                    thrust::device_vector<int> *nodes,
                    thrust::device_vector<int2> *leaves,
                    const int threshold,
                    const int max_level)
{
  const size_t num_points = points->size();
  /******************************************
   * 1. Generate points                     *
   ******************************************/

  // Generate random points using Thrust
  thrust::tabulate(points->begin(), points->end(), random_point());

  /******************************************
   * 2. Compute bounding box                *
   ******************************************/

  bbox bounds = compute_bounding_box(*points);

  /******************************************
   * 3. Classify points                     *
   ******************************************/

  thrust::device_vector<int> tags(num_points);
  
  compute_tags(*points, bounds, max_level, tags);

  /******************************************
   * 4. Sort according to classification    *
   ******************************************/

  thrust::device_vector<int> indices(num_points);

  // Now that we have the geometric information, we can sort the
  // points accordingly.
  sort_points_by_tag(tags, indices);

  /******************************************
   * 5. Build the tree                      *
   ******************************************/

  build_tree(tags, bounds, max_level, threshold, *nodes, *leaves);
}

int main()
{
  const size_t num_points = 4*1024*1024;
  const int threshold = 32; // A node with fewer than threshold points is a leaf.
  const int max_level = 10;

  thrust::device_vector<float2> points(num_points);
  thrust::device_vector<int> nodes;
  thrust::device_vector<int2> leaves;

  std::cout << "Warming up...\n" << std::endl;

  // validate and warm up the JIT
  run_experiment(&points, &nodes, &leaves, threshold, max_level);

  std::cout << "Timing...\n" << std::endl;

  size_t num_trials = 25;
  double mean_msecs = time_invocation_cuda(num_trials, run_experiment, &points, &nodes, &leaves, threshold, max_level);
  double mean_secs = mean_msecs / 1000;
  double millions_of_points = double(num_points) / 1000000;

  std::cout << millions_of_points / mean_secs << " millions of points generated and treeified per second." << std::endl;
}

