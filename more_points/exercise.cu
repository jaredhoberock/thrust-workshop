#include <thrust/device_vector.h>
#include <thrust/tabulate.h>
#include <thrust/random.h>
#include <thrust/transform.h>
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

bbox compute_bounding_box(const std::vector<float2> &points)
{
  std::cout << "TODO: compute the bounding box using thrust::reduce\n" << std::endl;
  bbox bounds;
  for (int i = 0 ; i < points.size() ; ++i)
  {
    float2 p = points[i];
    if ( p.x < bounds.xmin ) bounds.xmin = p.x;
    if ( p.x > bounds.xmax ) bounds.xmax = p.x;
    if ( p.y < bounds.ymin ) bounds.ymin = p.y;
    if ( p.y > bounds.ymax ) bounds.ymax = p.y;
  }

  return bounds;
}


void compute_tags(const std::vector<float2> &points, const bbox &bounds, int max_level, std::vector<int> &tags)
{
  std::cout << "TODO: classify the points using thrust::transform\n" << std::endl;
  for (int i = 0 ; i < points.size() ; ++i)
  {
    float2 p = points[i];
    tags[i] = point_to_tag(p, bounds, max_level);
  }
}


struct compare_tags
{
  template <typename pair_type>
  inline bool operator()(const pair_type &p0, const pair_type &p1) const
  {
    return p0.first < p1.first;
  }
};

void sort_points_by_tag(std::vector<int> &tags, std::vector<int> &indices)
{
  std::cout << "TODO: sort the points using thrust::sort_by_key\n" << std::endl;
  std::vector<std::pair<int, int> > tag_index_pairs(tags.size());
  for (int i = 0 ; i < tags.size() ; ++i)
  {
    tag_index_pairs[i].first  = tags[i];
    tag_index_pairs[i].second = i;
  }
  
  std::sort(tag_index_pairs.begin(), tag_index_pairs.end(), compare_tags());
  
  for (int i = 0 ; i < tags.size() ; ++i)
  {
    tags[i]    = tag_index_pairs[i].first;
    indices[i] = tag_index_pairs[i].second;
  }
}


void compute_child_tag_masks(const std::vector<int> &active_nodes,
                             int level,
                             size_t max_level,
                             std::vector<int> &children)
{
  std::cout << "TODO: compute child masks on GPU using thrust::transform\n";
  for (int i = 0 ; i < active_nodes.size() ; ++i)
  {
    int tag = active_nodes[i];
    children[4*i+0] = child_tag_mask(tag, 0, level, max_level);
    children[4*i+1] = child_tag_mask(tag, 1, level, max_level);
    children[4*i+2] = child_tag_mask(tag, 2, level, max_level);
    children[4*i+3] = child_tag_mask(tag, 3, level, max_level);
  }
}


void find_child_bounds(const std::vector<int> &tags,
                       const std::vector<int> &children,
                       int level,
                       size_t max_level,
                       std::vector<int> &lower_bounds,
                       std::vector<int> &upper_bounds)
{
  std::cout << "TODO: calculate bounds on the GPU using thrust::lower_bound and thrust::upper_bound\n";
  int length = (1 << (max_level - level) * 2) - 1;
  for (int i = 0 ; i < children.size() ; ++i)
  {
    lower_bounds[i] = (int)std::distance(tags.begin(),
                              std::lower_bound(tags.begin(), tags.end(), children[i]));
    
    upper_bounds[i] = (int)std::distance(tags.begin(),
                              std::upper_bound(tags.begin(), tags.end(), children[i] + length));
  }
}


void classify_children(const std::vector<int> &children,
                       const std::vector<int> &lower_bounds,
                       const std::vector<int> &upper_bounds,
                       int level,
                       int max_level,
                       int threshold,
                       std::vector<int> &child_node_kind)
{
  std::cout << "TODO: mark the children as nodes/leaves using thrust::transform\n";
  for(int i = 0 ; i < children.size() ; ++i )
  {
    int count = upper_bounds[i] - lower_bounds[i];
    if(count == 0)
    {
      child_node_kind[i] = EMPTY;
    }
    else if(level == max_level || count < threshold)
    {
      child_node_kind[i] = LEAF;
    }
    else
    {
      child_node_kind[i] = NODE;
    }
  }
}


// returns the number of nodes and leaves on this level, respectively
std::pair<int,int> enumerate_nodes_and_leaves(const std::vector<int> &child_node_kind,
                                              std::vector<int> &nodes_on_this_level,
                                              std::vector<int> &leaves_on_this_level)
{
  std::cout << "TODO: move the node enumeration to the GPU using thrust::transform_exclusive_scan\n";
  for(int i = 0, prefix_sum = 0 ; i < child_node_kind.size() ; ++i)
  {
    nodes_on_this_level[i] = prefix_sum;
    if(child_node_kind[i] == NODE)
    {
      ++prefix_sum;
    }
  }

  std::cout << "TODO: move the leaf enumeration to the GPU using thrust::transform_exclusive_scan\n";
  for(int i = 0, prefix_sum = 0 ; i < child_node_kind.size() ; ++i)
  {
    leaves_on_this_level[i] = prefix_sum;
    if(child_node_kind[i] == LEAF)
    {
      ++prefix_sum;
    }
  }

  std::pair<int,int> num_nodes_and_leaves_on_this_level;

  num_nodes_and_leaves_on_this_level.first = nodes_on_this_level.back() + (child_node_kind.back() == NODE ? 1 : 0);
  num_nodes_and_leaves_on_this_level.second = leaves_on_this_level.back() + (child_node_kind.back() == LEAF ? 1 : 0);

  return num_nodes_and_leaves_on_this_level;
}


void create_child_nodes(const std::vector<int> &child_node_kind,
                        const std::vector<int> &nodes_on_this_level,
                        const std::vector<int> &leaves_on_this_level,
                        int num_leaves,
                        std::vector<int> &nodes)
{
  int num_children = child_node_kind.size();

  int children_begin = nodes.size();
  nodes.resize(nodes.size() + num_children);
  
  std::cout << "TODO: add children to node list on the GPU using thrust::transform\n";
  for(int i = 0 ; i < num_children; ++i)
  {
    switch(child_node_kind[i])
    {
    case EMPTY:
      nodes[children_begin + i] = get_empty_id();
      break;
    case LEAF:
      nodes[children_begin + i] = get_leaf_id(num_leaves + leaves_on_this_level[i]);
      break;
    case NODE:
      nodes[children_begin + i] = children_begin + num_children + 4 * nodes_on_this_level[i];
      break;
    }
  }
}


void build_tree(const std::vector<int> &tags,
                const bbox &bounds,
                size_t max_level,
                int threshold,
                std::vector<int> &nodes,
                std::vector<int2> &leaves)
{
  std::cout << "TODO: move these active nodes to the GPU using thrust::device_vector\n" << std::endl;
  std::vector<int> active_nodes(1,0);

  int num_leaves = 0;

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
    std::cout << "TODO: move these children to the GPU using thrust::device_vector\n";
    std::vector<int> children(4*num_active_nodes);

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
    std::cout << "TODO: move these bounds to the GPU using thrust::device_vector\n";
    std::vector<int> lower_bounds(children.size());
    std::vector<int> upper_bounds(children.size());

    // Locate lower and upper bounds for points in each quadrant
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
    std::cout << "TODO: move these markers to the GPU using thrust::device_vector\n";
    std::vector<int> child_node_kind(children.size(), 0);

    classify_children(children, lower_bounds, upper_bounds, level, max_level, threshold, child_node_kind);

    std::cout << "child_node_kind:\n";
    for (int i = 0 ; i < children.size() ; ++i)
    {
      std::cout << std::setw(4) << i << ": [ ";
      std::cout << std::setw(5) << std::right;
      switch (child_node_kind[i])
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
    std::cout << "TODO: move these nodes and leaves to the GPU using thrust::device_vector\n";
    std::vector<int> nodes_on_this_level(child_node_kind.size());
    std::vector<int> leaves_on_this_level(child_node_kind.size());

    // Enumerate nodes and leaves at this level
    std::pair<int,int> num_nodes_and_leaves_on_this_level =
      enumerate_nodes_and_leaves(child_node_kind, nodes_on_this_level, leaves_on_this_level);

    std::cout << "Node/leaf enumeration:\n      [ nodeid leafid ]\n";
    for (int i = 0 ; i < children.size() ; ++i)
    {
      std::cout << std::setw(4) << i << ": [ ";
      switch (child_node_kind[i])
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

    create_child_nodes(child_node_kind, nodes_on_this_level, leaves_on_this_level, num_leaves, nodes);

    print_nodes(nodes);

    /******************************************
     * 6. Add the leaves to the leaf list     *
     ******************************************/

    // Add child leaves to the list of leaves
    leaves.resize(num_leaves + num_nodes_and_leaves_on_this_level.second);

    std::cout << "TODO: add child leaves to leaf list on the GPU using thrust::scatter_if\n";
    for(int i = 0 ; i < children.size() ; ++i)
    {
      if(child_node_kind[i] == LEAF)
      {
        leaves[num_leaves + leaves_on_this_level[i]] = make_int2(lower_bounds[i], upper_bounds[i]);
      }
    }

    // Update the number of leaves
    num_leaves += num_nodes_and_leaves_on_this_level.second;

    print_leaves(leaves);

    /******************************************
     * 7. Set the nodes for the next level    *
     ******************************************/
    
    // Set active nodes for the next level to be all the childs nodes from this level
    active_nodes.resize(num_nodes_and_leaves_on_this_level.first);

    std::cout << "TODO: add child nodes to next level on GPU using thrust::copy_if\n";
    for (int i = 0, j = 0 ; i < children.size() ; ++i)
    {
      if (child_node_kind[i] == NODE)
      {
        active_nodes[j++] = children[i];
      }
    }

    // Update the number of active nodes.
    num_active_nodes = num_nodes_and_leaves_on_this_level.first;
  }
}


int main()
{
  const size_t num_points = 12;
  const int threshold = 2; // A node with fewer than threshold points is a leaf.
  const int max_level = 3;

  std::vector<float2> points(num_points);

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

  std::cout << "TODO: move these tags to the GPU using thrust::device_vector\n" << std::endl;
  std::vector<int> tags(num_points);

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

  std::cout << "TODO: move these indices to the GPU using thrust::device_vector\n" << std::endl;
  std::vector<int> indices(num_points);

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

  std::cout << "TODO: move these nodes to the GPU using thrust::device_vector\n" << std::endl;
  std::vector<int> nodes;
  std::cout << "TODO: move these leaves to the GPU using thrust::device_vector\n" << std::endl;
  std::vector<int2> leaves;

  build_tree(tags, bounds, max_level, threshold, nodes, leaves);

  return 0;
}


