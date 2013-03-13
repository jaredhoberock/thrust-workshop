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

struct compare_tags
{
  template <typename pair_type>
  inline bool operator()(const pair_type &p0, const pair_type &p1) const
  {
    return p0.first < p1.first;
  }
};

struct bbox
{
  float xmin, xmax;
  float ymin, ymax;

  // TODO: annotate this function with __host__ __device__ so
  //       so that they are able to work with Thrust
  inline
  bbox() : xmin(FLT_MAX), xmax(-FLT_MAX), ymin(FLT_MAX), ymax(-FLT_MAX)
  {}
  
  // TODO: annotate this function with __host__ __device__ so
  //       so that they are able to work with Thrust
  inline
  bbox(const float2 &p) : xmin(p.x), xmax(p.x), ymin(p.y), ymax(p.y)
  {}
};

// Operator which merges two bounding boxes.
struct merge_bboxes
{
  // TODO: annotate this function with __host__ __device__ so
  //       so that they are able to work with Thrust
  inline
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

// Utility functions to encode leaves and children in single int
// are defined in util.h:
//   bool is_empty(int id);
//   bool is_node(int id);
//   bool is_leaf(int id);
//   int get_empty_id();
//   int get_leaf_id(int offset);
//   int get_leaf_offset(int id);

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

  std::cout << "TODO: compute the bounding box using thrust::reduce\n" << std::endl;
  bbox bounds;
  for (int i = 0 ; i < num_points ; ++i)
  {
    float2 p = points[i];
    if ( p.x < bounds.xmin ) bounds.xmin = p.x;
    if ( p.x > bounds.xmax ) bounds.xmax = p.x;
    if ( p.y < bounds.ymin ) bounds.ymin = p.y;
    if ( p.y > bounds.ymax ) bounds.ymax = p.y;
  }

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

  std::cout << "TODO: classify the points using thrust::transform\n" << std::endl;
  for (int i = 0 ; i < num_points ; ++i)
  {
    bbox box = bounds;
    float2 p = points[i];
    int tag = 0;

    for (int level = 1 ; level <= max_level ; ++level)
    {
      // Classify in x-direction
      float xmid = 0.5f * (box.xmin + box.xmax);
      int x_hi_half = (p.x < xmid) ? 0 : 1;

      // Push the bit into the tag as we build it
      tag |= x_hi_half;
      tag <<= 1;

      // Classify in y-direction
      float ymid = 0.5f * (box.ymin + box.ymax);
      int y_hi_half = (p.y < ymid) ? 0 : 1;

      // Push the bit into the tag as we build it
      tag |= y_hi_half;
      tag <<= 1;

      // Shrink the bounding box, still encapsulating the point
      box.xmin = (x_hi_half) ? xmid : box.xmin;
      box.xmax = (x_hi_half) ? box.xmax : xmid;
      box.ymin = (y_hi_half) ? ymid : box.ymin;
      box.ymax = (y_hi_half) ? box.ymax : ymid;
    }
    // Unshift the last
    tag >>= 1;
    tags[i] = tag;
  }

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
  std::cout << "TODO: sort the points using thrust::sort_by_key\n" << std::endl;
  std::vector<std::pair<int, int> > tag_index_pairs(num_points);
  for (int i = 0 ; i < num_points ; ++i)
  {
    tag_index_pairs[i].first  = tags[i];
    tag_index_pairs[i].second = i;
  }
  
  std::sort(tag_index_pairs.begin(), tag_index_pairs.end(), compare_tags());
  
  for (int i = 0 ; i < num_points ; ++i)
  {
    tags[i]    = tag_index_pairs[i].first;
    indices[i] = tag_index_pairs[i].second;
  }

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
  std::cout << "TODO: move these active nodes to the GPU using thrust::device_vector\n" << std::endl;
  std::vector<int> active_nodes(1,0);

  int num_nodes = 0, num_leaves = 0;

  // Build the tree one level at a time, starting at the root
  for (int level = 1 ; !active_nodes.empty() && level <= max_level ; ++level)
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
     * 6. Calculate children                  *
     ******************************************/

    // New children: 4 quadrants per active node = 4 children
    std::cout << "TODO: move these children to the GPU using thrust::device_vector\n";
    std::vector<int> children(4*num_active_nodes);

    // For each active node, generate the tag mask for each of its 4 children
    std::cout << "TODO: compute child masks on GPU using thrust::transform\n";
    int shift = (max_level - level) * 2;
    for (int i = 0 ; i < num_active_nodes ; ++i)
    {
      int tag = active_nodes[i];
      children[4*i+0] = tag | (0 << shift);
      children[4*i+1] = tag | (1 << shift);
      children[4*i+2] = tag | (2 << shift);
      children[4*i+3] = tag | (3 << shift);
    }

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
     * 7. Determine interval for each child   *
     ******************************************/

    // For each child we need interval bounds
    std::cout << "TODO: move these bounds to the GPU using thrust::device_vector\n";
    std::vector<int> lower_bounds(children.size());
    std::vector<int> upper_bounds(children.size());

    // Locate lower and upper bounds for points in each quadrant
    std::cout << "TODO: calculate bounds on the GPU using thrust::lower_bound and thrust::upper_bound\n";
    int length = (1 << (max_level - level) * 2) - 1;
    for (int i = 0 ; i < children.size() ; ++i)
    {
      lower_bounds[i] = (int)std::distance(tags.begin(),
                                std::lower_bound(tags.begin(), tags.end(), children[i]));
      
      upper_bounds[i] = (int)std::distance(tags.begin(),
                                std::upper_bound(tags.begin(), tags.end(), children[i] + length));
    }

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
     * 8. Mark each child as empty/leaf/node  *
     ******************************************/

    // Mark each child as either empty, a node, or a leaf
    std::cout << "TODO: move these markers to the GPU using thrust::device_vector\n";
    std::vector<int> markers(children.size(), 0);

    std::cout << "TODO: mark the children as nodes/leaves using thrust::transform\n";
    for (int i = 0 ; i < children.size() ; ++i )
    {
      int count = upper_bounds[i] - lower_bounds[i];
      if (count == 0)
      {
        markers[i] = EMPTY;
      }
      else if (level == max_level || count < threshold)
      {
        markers[i] = LEAF;
      }
      else
      {
        markers[i] = NODE;
      }
    }

    std::cout << "Child markers:\n";
    for (int i = 0 ; i < children.size() ; ++i)
    {
      std::cout << std::setw(4) << i << ": [ ";
      std::cout << std::setw(5) << std::right;
      switch (markers[i])
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
     * 9. Enumerate nodes and leaves          *
     ******************************************/

    // Enumerate the nodes and leaves at this level
    std::cout << "TODO: move these nodes and leaves to the GPU using thrust::device_vector\n";
    std::vector<int> level_nodes(markers.size());
    std::vector<int> level_leaves(markers.size());

    // Enumerate nodes at this level
    std::cout << "TODO: move the node emuration to the GPU using thrust::exclusive_scan\n";
    for (int i = 0, prefix_sum = 0 ; i < markers.size() ; ++i)
    {
      level_nodes[i] = prefix_sum;
      if (markers[i] == NODE)
      {
        ++prefix_sum;
      }
    }
    int num_level_nodes = level_nodes.back() + (markers.back() == NODE ? 1 : 0);

    // Enumerate leaves at this level
    std::cout << "TODO: move the leaf emuration to the GPU using thrust::exclusive_scan\n";
    for (int i = 0, prefix_sum = 0 ; i < markers.size() ; ++i)
    {
      level_leaves[i] = prefix_sum;
      if (markers[i] == LEAF)
      {
        ++prefix_sum;
      }
    }
    int num_level_leaves = level_leaves.back() + (markers.back() == LEAF ? 1 : 0);

    std::cout << "Node/leaf enumeration:\n      [ nodeid leafid ]\n";
    for (int i = 0 ; i < children.size() ; ++i)
    {
      std::cout << std::setw(4) << i << ": [ ";
      switch (markers[i])
      {
      case EMPTY:
        std::cout << std::setw(4) << "." << "   " << std::setw(4) << "." << "   ]";
        break;
      case LEAF:
        std::cout << std::setw(4) << "." << "   " << std::setw(4) << level_leaves[i] << "   ]";
        break;
      case NODE:
        std::cout << std::setw(4) << level_nodes[i] << "   " << std::setw(4) << "." << "   ]";
        break;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    /******************************************
     * 10. Add the children to the node list  *
     ******************************************/

    // Add these children to the list of nodes
    nodes.resize(num_nodes + children.size());

    std::cout << "TODO: add children to node list on the GPU using thrust::transform\n";
    for (int i = 0 ; i < children.size() ; ++i )
    {
      switch (markers[i])
      {
      case EMPTY:
        nodes[num_nodes + i] = get_empty_id();
        break;
      case LEAF:
        nodes[num_nodes + i] = get_leaf_id(num_leaves + level_leaves[i]);
        break;
      case NODE:
        nodes[num_nodes + i] = num_nodes + children.size() + 4 * level_nodes[i];
        break;
      }
    }

    // Update the number of nodes
    num_nodes += children.size();

    print_nodes(nodes);

    /******************************************
     * 11. Add the leaves to the leaf list    *
     ******************************************/

    // Add child leaves to the list of leaves
    leaves.resize(num_leaves + num_level_leaves);

    std::cout << "TODO: add child leaves to leaf list on the GPU using thrust::scatter_if\n";
    for (int i = 0 ; i < children.size() ; ++i)
    {
      if (markers[i] == LEAF)
      {
        leaves[num_leaves + level_leaves[i]] = make_int2(lower_bounds[i], upper_bounds[i]);
      }
    }

    // Update the number of leaves
    num_leaves += num_level_leaves;

    print_leaves(leaves);

    /******************************************
     * 12. Set the nodes for the next level   *
     ******************************************/
    
    // Set active nodes for the next level to be all the childs nodes from this level
    active_nodes.resize(num_level_nodes);

    std::cout << "TODO: add child nodes to next level on GPU using thrust::copy_if\n";
    for (int i = 0, j = 0 ; i < children.size() ; ++i)
    {
      if (markers[i] == NODE)
      {
        active_nodes[j++] = children[i];
      }
    }

    // Update the number of active nodes.
    num_active_nodes = num_level_nodes;
  }

  return 0;
}


