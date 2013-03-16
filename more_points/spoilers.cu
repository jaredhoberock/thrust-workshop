#include <thrust/device_vector.h>
#include <thrust/tabulate.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <iostream>
#include <iomanip>
#include <bitset>
#include <algorithm>
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
//   int child_tag_mask(int tag, int which_child, int level, int max_level);

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
  thrust::device_ptr<const int> nodes;
  
  expand_active_nodes(int lvl, int max_lvl, thrust::device_ptr<const int> nodes) : level(lvl), max_level(max_lvl), nodes(nodes) {}
  
  inline __device__ __host__
  int operator()(int idx) const
  {
    int tag = nodes[idx/4];
    int which_child = (idx&3);
    return child_tag_mask(tag, which_child, level, max_level);
  }
};


void compute_child_tag_masks(const thrust::device_vector<int> &active_nodes,
                             int level,
                             size_t max_level,
                             thrust::device_vector<int> &children)
{
  // For each active node, generate the tag mask for each of its 4 children
  thrust::transform(thrust::make_counting_iterator<int>(0),
                    thrust::make_counting_iterator<int>(children.size()),
                    children.begin(),
                    expand_active_nodes(level, max_level, active_nodes.data()));
}


void find_child_bounds(const thrust::device_vector<int> &tags,
                       const thrust::device_vector<int> &children,
                       int level,
                       size_t max_level,
                       thrust::device_vector<int> &lower_bounds,
                       thrust::device_vector<int> &upper_bounds)
{
  // Locate lower and upper bounds for points in each quadrant
  thrust::lower_bound(tags.begin(),
                      tags.end(),
                      children.begin(),
                      children.end(),
                      lower_bounds.begin());
  
  int length = (1 << (max_level - level) * 2) - 1;

  using namespace thrust::placeholders;

  thrust::upper_bound(tags.begin(),
                      tags.end(),
                      thrust::make_transform_iterator(children.begin(), _1 + length),
                      thrust::make_transform_iterator(children.end(), _1 + length),
                      upper_bounds.begin());
}


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


void classify_children(const thrust::device_vector<int> &children,
                       const thrust::device_vector<int> &lower_bounds,
                       const thrust::device_vector<int> &upper_bounds,
                       int level,
                       int max_level,
                       int threshold,
                       thrust::device_vector<int> &child_node_kind)
{
  thrust::transform(thrust::make_zip_iterator(
                        thrust::make_tuple(lower_bounds.begin(), upper_bounds.begin())),
                    thrust::make_zip_iterator(
                        thrust::make_tuple(lower_bounds.end(), upper_bounds.end())),
                    child_node_kind.begin(),
                    mark_nodes(threshold, level == max_level));
}


std::pair<int,int> enumerate_nodes_and_leaves(const thrust::device_vector<int> &child_node_kind,
                                              thrust::device_vector<int> &nodes_on_this_level,
                                              thrust::device_vector<int> &leaves_on_this_level)
{
  // Enumerate nodes at this level
  thrust::transform_exclusive_scan(child_node_kind.begin(), 
                                   child_node_kind.end(), 
                                   nodes_on_this_level.begin(), 
                                   is_a<NODE>(), 
                                   0, 
                                   thrust::plus<int>());
  
  // Enumerate leaves at this level
  thrust::transform_exclusive_scan(child_node_kind.begin(), 
                                   child_node_kind.end(), 
                                   leaves_on_this_level.begin(), 
                                   is_a<LEAF>(), 
                                   0, 
                                   thrust::plus<int>());

  std::pair<int,int> num_nodes_and_leaves_on_this_level;

  num_nodes_and_leaves_on_this_level.first = nodes_on_this_level.back() + (child_node_kind.back() == NODE ? 1 : 0);
  num_nodes_and_leaves_on_this_level.second = leaves_on_this_level.back() + (child_node_kind.back() == LEAF ? 1 : 0);

  return num_nodes_and_leaves_on_this_level;
}


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


void create_child_nodes(const thrust::device_vector<int> &child_node_kind,
                        const thrust::device_vector<int> &nodes_on_this_level,
                        const thrust::device_vector<int> &leaves_on_this_level,
                        int num_leaves,
                        thrust::device_vector<int> &nodes)
{
  int num_children = child_node_kind.size();

  int children_begin = nodes.size();
  nodes.resize(nodes.size() + num_children);
  
  thrust::transform(thrust::make_zip_iterator(
                        thrust::make_tuple(
                            child_node_kind.begin(), nodes_on_this_level.begin(), leaves_on_this_level.begin())),
                    thrust::make_zip_iterator(
                        thrust::make_tuple(
                            child_node_kind.end(), nodes_on_this_level.end(), leaves_on_this_level.end())),
                    nodes.begin() + children_begin,
                    write_nodes(nodes.size(), num_leaves));
}


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


void create_leaves(const thrust::device_vector<int> &child_node_kind,
                   const thrust::device_vector<int> &leaves_on_this_level,
                   const thrust::device_vector<int> &lower_bounds,
                   const thrust::device_vector<int> &upper_bounds,
                   int num_leaves_on_this_level,
                   thrust::device_vector<int2> &leaves)
{
  int children_begin = leaves.size();

  leaves.resize(leaves.size() + num_leaves_on_this_level);

  thrust::scatter_if(thrust::make_transform_iterator(
                         thrust::make_zip_iterator(
                             thrust::make_tuple(lower_bounds.begin(), upper_bounds.begin())),
                         make_leaf()),
                     thrust::make_transform_iterator(
                         thrust::make_zip_iterator(
                             thrust::make_tuple(lower_bounds.end(), upper_bounds.end())),
                         make_leaf()),
                     leaves_on_this_level.begin(),
                     child_node_kind.begin(),
                     leaves.begin() + children_begin,
                     is_a<LEAF>());
}


void activate_nodes_for_next_level(const thrust::device_vector<int> &children,
                                   const thrust::device_vector<int> &child_node_kind,
                                   int num_nodes_on_this_level,
                                   thrust::device_vector<int> &active_nodes)
{
  // Set active nodes for the next level to be all the childs nodes from this level
  active_nodes.resize(num_nodes_on_this_level);
  
  thrust::copy_if(children.begin(),
                  children.end(),
                  child_node_kind.begin(),
                  active_nodes.begin(),
                  is_a<NODE>());
}


void build_tree(const thrust::device_vector<int> &tags,
                const bbox &bounds,
                size_t max_level,
                int threshold,
                thrust::device_vector<int> &nodes,
                thrust::device_vector<int2> &leaves)
{
  thrust::device_vector<int> active_nodes(1,0);

  // Build the tree one level at a time, starting at the root
  for(int level = 1 ; !active_nodes.empty() && level <= max_level ; ++level)
  {
    std::cout << "\n\n\n*************************\n";
    std::cout << "*** BUILDING LEVEL " << std::setw(4) << level << " *\n";
    std::cout << "*************************\n";

    // Number of nodes to process at this level
    int num_active_nodes = static_cast<int>(active_nodes.size());

    std::cout << "Active nodes:\n      ";
    for (int i = 1 ; i <= max_level ; ++i)
    {
      std::cout << "xy ";
    }
    std::cout << std::endl;
    for (int i = 0 ; i < num_active_nodes ; ++i)
    {
      std::cout << std::setw(4) << i << ": ";
      print_tag(active_nodes[i], max_level);
      std::cout << std::endl;
    }
    std::cout << std::endl;

    /******************************************
     * 1. Calculate children                  *
     ******************************************/

    // New children: 4 quadrants per active node = 4 children
    thrust::device_vector<int> children(4*num_active_nodes);

    compute_child_tag_masks(active_nodes, level, max_level, children);

    std::cout << "Children:\n      ";
    for (int i = 1 ; i <= max_level ; ++i)
    {
      std::cout << "xy ";
    }
    std::cout << std::endl;
    for (int i = 0 ; i < children.size() ; ++i)
    {
      std::cout << std::setw(4) << i << ": ";
      print_tag(children[i], max_level);
      std::cout << std::endl;
    }
    std::cout << std::endl;

    /******************************************
     * 2. Determine interval for each child   *
     ******************************************/

    // For each child we need interval bounds
    thrust::device_vector<int> lower_bounds(children.size());
    thrust::device_vector<int> upper_bounds(children.size());

    find_child_bounds(tags, children, level, max_level, lower_bounds, upper_bounds);

    std::cout << "Child bounds:\n      [ lower upper count ]\n";
    for (int i = 0 ; i < children.size() ; ++i)
    {
      std::cout << std::setw(4) << i << ": [ ";
      std::cout << std::setw(4) << lower_bounds[i] << "  ";
      std::cout << std::setw(4) << upper_bounds[i] << "  ";
      std::cout << std::setw(4) << upper_bounds[i] - lower_bounds[i] << "  ]";
      std::cout << std::endl;
    }
    std::cout << std::endl;

    /******************************************
     * 3. Mark each child as empty/leaf/node  *
     ******************************************/

    // Mark each child as either empty, a node, or a leaf
    thrust::device_vector<int> child_node_kind(children.size(), 0);
    classify_children(children, lower_bounds, upper_bounds, level, max_level, threshold, child_node_kind);

    std::cout << "child_node_kind:\n";
    for (int i = 0 ; i < children.size() ; ++i)
    {
      std::cout << std::setw(4) << i << ": [ ";
      std::cout << std::setw(5) << std::right;
      switch(child_node_kind[i])
      {
      case EMPTY:
        std::cout << "EMPTY ]";
        break;
      case LEAF:
        std::cout << "LEAF ]";
        break;
      case NODE:
        std::cout << "NODE ]";
        break;
      default:
        std::cout << "ERROR ]";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    /******************************************
     * 4. Enumerate nodes and leaves          *
     ******************************************/

    // Enumerate the nodes and leaves at this level
    thrust::device_vector<int> leaves_on_this_level(child_node_kind.size());
    thrust::device_vector<int> nodes_on_this_level(child_node_kind.size());

    // Enumerate nodes and leaves at this level
    std::pair<int,int> num_nodes_and_leaves_on_this_level =
      enumerate_nodes_and_leaves(child_node_kind, nodes_on_this_level, leaves_on_this_level);

    std::cout << "Node/leaf enumeration:\n      [ nodeid leafid ]\n";
    for(int i = 0 ; i < children.size() ; ++i)
    {
      std::cout << std::setw(4) << i << ": [ ";
      switch(child_node_kind[i])
      {
      case EMPTY:
        std::cout << std::setw(4) << "." << "   " << std::setw(4) << "." << "   ]";
        break;
      case LEAF:
        std::cout << std::setw(4) << "." << "   " << std::setw(4) << leaves_on_this_level[i] << "   ]";
        break;
      case NODE:
        std::cout << std::setw(4) << nodes_on_this_level[i] << "   " << std::setw(4) << "." << "   ]";
        break;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    /******************************************
     * 5. Add the children to the node list   *
     ******************************************/

    create_child_nodes(child_node_kind, nodes_on_this_level, leaves_on_this_level, leaves.size(), nodes);

    print_nodes(nodes);

    /******************************************
     * 6. Add the leaves to the leaf list     *
     ******************************************/

    create_leaves(child_node_kind, leaves_on_this_level, lower_bounds, upper_bounds, num_nodes_and_leaves_on_this_level.second, leaves);

    print_leaves(leaves);

    /******************************************
     * 7. Set the nodes for the next level    *
     ******************************************/

    activate_nodes_for_next_level(children, child_node_kind, num_nodes_and_leaves_on_this_level.first, active_nodes);
    
    // Update the number of active nodes.
    num_active_nodes = num_nodes_and_leaves_on_this_level.first;
  }
}

int main()
{
  const size_t num_points = 12;
  const int threshold = 2; // A node with fewer than threshold points is a leaf.
  const int max_level = 3;

  thrust::device_vector<float2> points(num_points);

  /******************************************
   * 1. Generate points                     *
   ******************************************/

  // Generate random points using Thrust
  thrust::tabulate(points.begin(), points.end(), random_point());

  std::cout << "Points:\n";
  for (int i = 0 ; i < points.size() ; ++i)
  {
    std::cout << std::setw(4) << i << " " << points[i] << std::endl;
  }
  std::cout << std::endl;

  /******************************************
   * 2. Compute bounding box                *
   ******************************************/

  bbox bounds = compute_bounding_box(points);

  float xmid = 0.5f * (bounds.xmin + bounds.xmax);
  float ymid = 0.5f * (bounds.ymin + bounds.ymax);
  std::cout << "Bounding box:\n";
  std::cout << "   min: " << make_float2(bounds.xmin, bounds.ymin) << std::endl;
  std::cout << "   mid: " << make_float2(xmid, ymid) << std::endl;
  std::cout << "   max: " << make_float2(bounds.xmax, bounds.ymax) << std::endl;
  std::cout << std::endl;

  /******************************************
   * 3. Classify points                     *
   ******************************************/

  thrust::device_vector<int> tags(num_points);
  
  compute_tags(points, bounds, max_level, tags);

  std::cout << "Tags:                       ";
  for (int level = 1 ; level <= max_level ; ++level)
  {
    std::cout << std::setw(3) << std::left << level;
  }
  std::cout << "\n                            ";
  for (int level = 1 ; level <= max_level ; ++level)
  {
    std::cout << "xy ";
  }
  std::cout << std::right << std::endl;
  for (int i = 0 ; i < points.size() ; ++i)
  {
    int tag = tags[i];
    std::cout << std::setw(4) << i << " " << points[i] << ":  ";
    print_tag(tags[i], max_level);
    std::cout << std::endl;
  }
  std::cout << std::endl;
  
  /******************************************
   * 4. Sort according to classification    *
   ******************************************/

  thrust::device_vector<int> indices(num_points);

  // Now that we have the geometric information, we can sort the
  // points accordingly.
  sort_points_by_tag(tags, indices);

  std::cout << "Sorted tags:                ";
  for (int level = 1 ; level <= max_level ; ++level)
  {
    std::cout << std::setw(3) << std::left << level;
  }
  std::cout << "\n                            ";
  for (int level = 1 ; level <= max_level ; ++level)
  {
    std::cout << "xy ";
  }
  std::cout << std::right << std::endl;
  for (int i = 0 ; i < points.size() ; ++i)
  {
    int tag = tags[i];
    std::cout << std::setw(4) << i << " " << points[i] << ":  ";
    print_tag(tags[i], max_level);
    std::cout << "  original index " << std::setw(4) << indices[i] << std::endl;
  }
  std::cout << std::endl;

  /******************************************
   * 5. Build the tree                      *
   ******************************************/

  thrust::device_vector<int> nodes;
  thrust::device_vector<int2> leaves;
  
  build_tree(tags, bounds, max_level, threshold, nodes, leaves);

  return 0;
}

