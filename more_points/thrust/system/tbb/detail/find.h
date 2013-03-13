/*
 *  Copyright 2008-2012 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/detail/generic/find.h>
#include <thrust/system/tbb/detail/tag.h>

namespace thrust
{
namespace system
{
namespace tbb
{
namespace detail
{

template <typename System, typename InputIterator, typename Predicate>
InputIterator find_if(dispatchable<System> &system,
                      InputIterator first,
                      InputIterator last,
                      Predicate pred)
{
  // tbb prefers generic::find_if to cpp::find_if
  return thrust::system::detail::generic::find_if(system, first, last, pred);
}

} // end namespace detail
} // end namespace tbb
} // end namespace system
} // end namespace thrust
