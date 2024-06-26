// Copyright (c) 2024, DeepLink.
//
// Main entry for DIPU initialization and resource management.
//
// TODO(lljbash): refactor resource management code, build a global context

#include "DIPUGlobals.h"

#include <ctime>
#include <iostream>
#include <mutex>
#include <string>

#include "csrc_dipu/aten/OpRegister.hpp"
#include "csrc_dipu/runtime/core/DIPUEventPool.h"
#include "csrc_dipu/runtime/core/DIPUGeneratorImpl.h"
#include "csrc_dipu/runtime/core/allocator/DIPUCachingAllocatorUtils.h"
#include "csrc_dipu/runtime/devproxy/deviceproxy.h"

namespace dipu {

namespace {

void printPromptAtStartup() {
  auto time = std::time(nullptr);
  std::string time_str = std::ctime(&time);
  std::cout << time_str.substr(0, time_str.size() - 1)
            << " dipu | git hash:" << DIPU_GIT_HASH << std::endl;
}

void initResourceImpl() {
  static bool called(false);
  if (called) {
    return;
  }
  called = true;

  static std::once_flag print_prompt_flag;
  std::call_once(print_prompt_flag, printPromptAtStartup);

  devproxy::initializeVendor();
  initCachedAllocator();

  at::DipuOpRegister::instance().applyDelayedRegister();
}

void releaseAllResourcesImpl() {
  static bool called(false);
  if (called) {
    return;
  }
  called = true;
  releaseAllGenerator();
  releaseAllDeviceMem();
  releaseAllEvent();
  devproxy::finalizeVendor();
}

class DIPUIniter {
 public:
  DIPUIniter() { initResourceImpl(); }

  ~DIPUIniter() { releaseAllResourcesImpl(); }
};

}  // namespace

void initResource() {
  initResourceImpl();
  /* In some cases(eg: spawn process), the resource cleanup function we
     registered will not be executed, so we use the destructor of the static
     variable in the function here just in case. */
  static DIPUIniter initer;
}

void releaseAllResources() { releaseAllResourcesImpl(); }

}  // namespace dipu
