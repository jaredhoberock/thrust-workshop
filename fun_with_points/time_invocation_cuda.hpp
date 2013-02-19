#pragma once
#include <cstddef>
#include <cuda_runtime_api.h>

template<typename Function>
  double time_invocation_cuda(std::size_t num_trials, Function f)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f();
  }
  cudaEventRecord(stop);
  cudaThreadSynchronize();

  float msecs = 0;
  cudaEventElapsedTime(&msecs, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // return mean msecs
  return msecs / num_trials;
}

template<typename Function, typename Arg1>
  double time_invocation_cuda(std::size_t num_trials, Function f, Arg1 arg1)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f(arg1);
  }
  cudaEventRecord(stop);
  cudaThreadSynchronize();

  float msecs = 0;
  cudaEventElapsedTime(&msecs, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // return mean msecs
  return msecs / num_trials;
}

template<typename Function, typename Arg1, typename Arg2>
  double time_invocation_cuda(std::size_t num_trials, Function f, Arg1 arg1, Arg2 arg2)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f(arg1,arg2);
  }
  cudaEventRecord(stop);
  cudaThreadSynchronize();

  float msecs = 0;
  cudaEventElapsedTime(&msecs, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // return mean msecs
  return msecs / num_trials;
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3>
  double time_invocation_cuda(std::size_t num_trials, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f(arg1,arg2,arg3);
  }
  cudaEventRecord(stop);
  cudaThreadSynchronize();

  float msecs = 0;
  cudaEventElapsedTime(&msecs, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // return mean msecs
  return msecs / num_trials;
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
  double time_invocation_cuda(std::size_t num_trials, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f(arg1,arg2,arg3,arg4);
  }
  cudaEventRecord(stop);
  cudaThreadSynchronize();

  float msecs = 0;
  cudaEventElapsedTime(&msecs, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // return mean msecs
  return msecs / num_trials;
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
  double time_invocation_cuda(std::size_t num_trials, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f(arg1,arg2,arg3,arg4,arg5);
  }
  cudaEventRecord(stop);
  cudaThreadSynchronize();

  float msecs = 0;
  cudaEventElapsedTime(&msecs, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // return mean msecs
  return msecs / num_trials;
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6>
  double time_invocation_cuda(std::size_t num_trials, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f(arg1,arg2,arg3,arg4,arg5,arg6);
  }
  cudaEventRecord(stop);
  cudaThreadSynchronize();

  float msecs = 0;
  cudaEventElapsedTime(&msecs, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // return mean msecs
  return msecs / num_trials;
}

template<typename Function, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5, typename Arg6, typename Arg7>
  double time_invocation_cuda(std::size_t num_trials, Function f, Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f(arg1,arg2,arg3,arg4,arg5,arg6,arg7);
  }
  cudaEventRecord(stop);
  cudaThreadSynchronize();

  float msecs = 0;
  cudaEventElapsedTime(&msecs, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // return mean msecs
  return msecs / num_trials;
}


