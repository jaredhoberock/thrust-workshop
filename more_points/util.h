// Utility functions to encode leaves and children in single int
inline __device__ __host__
bool is_empty(int id) { return id == 0xffffffff; }

inline __device__ __host__
bool is_node(int id) { return id > 0; }

inline __device__ __host__
bool is_leaf(int id) { return id < 0; }

inline __device__ __host__
int get_empty_id() { return 0xffffffff; }

inline __device__ __host__
int get_leaf_id(int offset) { return 0x80000000 | offset; }

inline __device__ __host__
int get_leaf_offset(int id) { return 0x80000000 ^ id; }

template <int CODE>
struct is_a
{
  typedef int result_type;
  inline __device__ __host__
  int operator()(int code) { return code == CODE ? 1 : 0; }
};


// Given an integer, output a pseudorandom 2D point
struct random_point
{
  __host__ __device__ unsigned int hash(unsigned int x)
  {
    x = (x+0x7ed55d16) + (x<<12);
    x = (x^0xc761c23c) ^ (x>>19);
    x = (x+0x165667b1) + (x<<5);
    x = (x+0xd3a2646c) ^ (x<<9);
    x = (x+0xfd7046c5) + (x<<3);
    x = (x^0xb55a4f09) ^ (x>>16);
    return x;
  }

  __host__ __device__ float2 operator()(unsigned int i)
  {
    thrust::default_random_engine rng(hash(i));
    thrust::random::uniform_real_distribution<float> dist;
    float x = dist(rng);
    float y = dist(rng);
    return make_float2(x, y);
  }
};

struct bbox
{
  float xmin, xmax;
  float ymin, ymax;

  inline __host__ __device__
  bbox() : xmin(FLT_MAX), xmax(-FLT_MAX), ymin(FLT_MAX), ymax(-FLT_MAX)
  {}
  
  inline __host__ __device__
  bbox(const float2 &p) : xmin(p.x), xmax(p.x), ymin(p.y), ymax(p.y)
  {}
};

__host__ __device__
int point_to_tag(const float2 &p, bbox box, int max_level)
{
  int result = 0;
  
  for (int level = 1 ; level <= max_level ; ++level)
  {
    // Classify in x-direction
    float xmid = 0.5f * (box.xmin + box.xmax);
    int x_hi_half = (p.x < xmid) ? 0 : 1;
  
    // Push the bit into the result as we build it
    result |= x_hi_half;
    result <<= 1;
  
    // Classify in y-direction
    float ymid = 0.5f * (box.ymin + box.ymax);
    int y_hi_half = (p.y < ymid) ? 0 : 1;
  
    // Push the bit into the result as we build it
    result |= y_hi_half;
    result <<= 1;
  
    // Shrink the bounding box, still encapsulating the point
    box.xmin = (x_hi_half) ? xmid : box.xmin;
    box.xmax = (x_hi_half) ? box.xmax : xmid;
    box.ymin = (y_hi_half) ? ymid : box.ymin;
    box.ymax = (y_hi_half) ? box.ymax : ymid;
  }
  // Unshift the last
  result >>= 1;

  return result;
}

std::ostream &operator<<(std::ostream &os, float2 p)
{
  return os << std::fixed << "(" <<
      std::setw(8) << std::setprecision(6) << p.x << ", " <<
      std::setw(8) << std::setprecision(6) << p.y << ")";
}

void print_tag(int tag, int max_level)
{
  for (int level = 1 ; level <= max_level ; ++level)
  {
    std::bitset<2> bits = tag >> (max_level - level) * 2;
    std::cout << bits << " ";
  }
}

void print_nodes(const thrust::host_vector<int> &nodes)
{
  std::cout << "Quadtree nodes:\n";
  std::cout << "          [ nodeid  leafid ]\n";
  
  int next_level = 0;
  int children_at_next_level = 4;

  for (int i = 0 ; i < nodes.size() ; ++i)
  {
    if (i == next_level)
    {
      std::cout << "          [================]\n";
      next_level += children_at_next_level;
      children_at_next_level = 0;
    }
    else if (i % 4 == 0)
    {
      std::cout << "          [----------------]\n";
    }

    if (is_empty(nodes[i]))
    {
      std::cout << std::setw(7) << i << " : [ ";
      std::cout << std::setw(4) << "." << "    ";
      std::cout << std::setw(4) << "." << "   ]\n";
    }
    else if (is_leaf(nodes[i]))
    {
      std::cout << std::setw(7) << i << " : [ ";
      std::cout << std::setw(4) << "." << "    ";
      std::cout << std::setw(4) << get_leaf_offset(nodes[i]) << "   ]\n";
    }
    else
    {
      std::cout << std::setw(7) << i << " : [ ";
      std::cout << std::setw(4) << nodes[i] << "    ";
      std::cout << std::setw(4) << "." << "   ]\n";
    }
  }
  std::cout << "          [================]\n";
}

void print_leaves(const thrust::host_vector<int2> &leaves)
{
  std::cout << "Quadtree leaves:\n";
  std::cout << "          [ lower    upper ]\n";
  
  for (int i = 0 ; i < leaves.size() ; ++i)
  {
    std::cout << std::setw(7) << i << " : [ ";
    std::cout << std::setw(4) << leaves[i].x << "    ";
    std::cout << std::setw(4) << leaves[i].y << "   ]\n";
  }
}

