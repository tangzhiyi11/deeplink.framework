// Copyright (c) 2023, DeepLink.
#pragma once

#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/Stream.h>
#include <c10/util/Exception.h>

#include "csrc_dipu/aten/DIPUATenFunctions.h"
#include "csrc_dipu/aten/ops/OpUtils.hpp"
#include "csrc_dipu/profiler/profiler.h"
#include "csrc_dipu/runtime/core/DIPUEvent.h"
#include "csrc_dipu/runtime/core/DIPUGuard.h"
#include "csrc_dipu/runtime/core/DIPUStream.h"
#include "csrc_dipu/runtime/core/allocator/DIPUCachingAllocator.h"
#include "csrc_dipu/runtime/core/allocator/DIPUCachingAllocatorUtils.h"
#include "csrc_dipu/runtime/core/allocator/DIPUCachingHostAllocator.h"
#include "csrc_dipu/runtime/rthelper.h"
#include "csrc_dipu/utils/helpfunc.hpp"

namespace dipu {
namespace native {
// NOTICE: these 2 func defined in AutoGenedKernels.cpp
// if dipu autogen support header file gen, remove this
at::Tensor dipu_wrap_diopi_cast_dtype(const at::Tensor& self,
                                      at::ScalarType dtype);

// if dipu autogen support proxy one torch op to multiple diopi op, remove
// this.
at::Tensor& dipu_wrap_diopi_copy_inp(at::Tensor& self, const at::Tensor& src,
                                     bool non_blocking);

}  // namespace native

enum class DIPUCopyType {
  // src and dest tensor in one device
  D2Self,
  // from one device to another device.Not fully tested
  D2OtherD,
  // from device to host
  D2H,
  // from host to device
  H2D,
  // from host to host
  H2H,
};

// Align with pytorch's behavior, see TensorIterator.cpp compute_mem_overlaps()
inline void checkOverlap(const at::Tensor& dst, const at::Tensor& src) {
#if DIPU_TORCH_VERSION == 20000
  assert_no_internal_overlap(dst);
#else
  // seems torch2.1.0 (20100) not check internal overlap
#endif

  assert_no_partial_overlap(dst, src);
}

inline DIPUCopyType getCopyType(const at::Tensor& dst, const at::Tensor& src) {
  bool isSrcDevice = dipu::isDeviceTensor(src);
  bool isDstDevice = dipu::isDeviceTensor(dst);
  if (!isSrcDevice && isDstDevice) {
    return DIPUCopyType::H2D;
  }
  if (!isDstDevice && isSrcDevice) {
    return DIPUCopyType::D2H;
  }
  if (isSrcDevice && isDstDevice &&
      src.device().index() != dst.device().index()) {
    return DIPUCopyType::D2OtherD;
  }
  if (isSrcDevice && isDstDevice &&
      src.device().index() == dst.device().index()) {
    return DIPUCopyType::D2Self;
  }
  if (!isSrcDevice && !isDstDevice) {
    return DIPUCopyType::H2H;
  }
  return DIPUCopyType::H2H;
}

inline int64_t getMemCopyBytes(const at::Tensor& dst, const at::Tensor& src,
                               bool nonOverlappingAndDense) {
  if (dst.nbytes() !=
      src.nbytes()) {  // outer bytes must same. different type is unsuported
    TORCH_CHECK(false, "mem copy with different tensor size is not allowed");
  }
  if (nonOverlappingAndDense) {
    return static_cast<int64_t>(dst.nbytes());
  }
  auto dstBytes = static_cast<int64_t>(
      dst.unsafeGetTensorImpl()->unsafe_storage().nbytes());
  auto srcBytes = static_cast<int64_t>(
      src.unsafeGetTensorImpl()->unsafe_storage().nbytes());
  return std::min(srcBytes, dstBytes);
}

inline void doMemCopyH2D(const at::Tensor& dst, const at::Tensor& src,
                         dipu::DIPUStream& stream, int64_t nbytes,
                         bool isSynchronousCopy) {
  void* src_ptr = src.data_ptr();
  void* dst_ptr = dst.data_ptr();

  MemChecker::instance().check(dst);
  if (isSynchronousCopy) {
    dipu::devproxy::memCopyH2D(nbytes, dst_ptr, src_ptr);
  } else {
    dipu::devproxy::memCopyH2DAsync(stream.rawstream(), nbytes, dst_ptr,
                                    src_ptr);
  }
}

inline void doMemCopyD2H(const at::Tensor& dst, const at::Tensor& src,
                         dipu::DIPUStream& stream, int64_t nbytes,
                         bool isSynchronousCopy) {
  void* src_ptr = src.data_ptr();
  void* dst_ptr = dst.data_ptr();

  MemChecker::instance().check(src);
  if (isSynchronousCopy) {
    dipu::devproxy::memCopyD2H(nbytes, dst_ptr, src_ptr);
  } else {
    dipu::devproxy::memCopyD2HAsync(stream.rawstream(), nbytes, dst_ptr,
                                    src_ptr);
  }
}

inline void doMemCopyH2H(const at::Tensor& dst, const at::Tensor& src,
                         int64_t nbytes) {
  if (!dst.is_contiguous() || !src.is_contiguous()) {
    std::cerr << "Tensors must be contiguous for memory copy." << std::endl;
    return;
  }
  void* src_ptr = src.data_ptr();
  void* dst_ptr = dst.data_ptr();
  memcpy(dst_ptr, src_ptr, nbytes);
}

inline void doMemCopyD2D(const at::Tensor& dst, const at::Tensor& src,
                         dipu::DIPUStream& stream, int64_t nbytes,
                         bool isSynchronousCopy) {
  dipu::devproxy::memCopyD2DAsync(stream.rawstream(), nbytes,
                                  dst.device().index(), dst.data_ptr(),
                                  src.device().index(), src.data_ptr());
}

inline void memCopy(const at::Tensor& dst, const at::Tensor& src,
                    dipu::DIPUStream& stream, DIPUCopyType copyType,
                    bool nonOverlappingAndDense, bool isSynchronousCopy) {
  int64_t nbytes = getMemCopyBytes(dst, src, nonOverlappingAndDense);
  switch (copyType) {
    case DIPUCopyType::H2D:
      // src is cpu.
      doMemCopyH2D(dst, src, stream, nbytes, isSynchronousCopy);
      break;
    case DIPUCopyType::D2H:
      // dst is cpu.
      doMemCopyD2H(dst, src, stream, nbytes, isSynchronousCopy);
      break;
    case DIPUCopyType::H2H:
      doMemCopyH2H(dst, src, nbytes);
      break;
    default:  // device to device
      doMemCopyD2D(dst, src, stream, nbytes, isSynchronousCopy);
  }
}

class CopyParamsInfo {
 public:
  DIPUCopyType copyType_;
  DIPUStream curStream_;
  // basic info
  // if cast needed
  bool sameDtype_ = false;
  // determine if expand needed.
  bool sameSize_ = false;
  bool sameStride_ = false;
  bool denseAndNoOverlap_ = false;
  c10::DeviceIndex srcDevice_ = -1;
  c10::DeviceIndex dstDevice_ = -1;

  // composite info, can direct mem copy
  bool directMemCopy_ = false;

  void recomputeTensorsInfo(const at::Tensor& dst, const at::Tensor& src) {
    sameDtype_ = dst.scalar_type() == src.scalar_type();
    sameSize_ = dst.sizes().equals(src.sizes());
    sameStride_ = dst.strides().equals(src.strides());
    denseAndNoOverlap_ = dst.is_non_overlapping_and_dense() &&
                         src.is_non_overlapping_and_dense();
    directMemCopy_ =
        sameDtype_ && sameSize_ && sameStride_ && denseAndNoOverlap_;
    srcDevice_ = src.device().index();
    dstDevice_ = dst.device().index();
  }

  explicit CopyParamsInfo(const at::Tensor& dst, const at::Tensor& src,
                          const DIPUStream& curStream) {
    // assume layout always = not suppport Sparse layout
    TORCH_CHECK(dst.options().layout() == c10::Layout::Strided,
                "only Strided layout is supported");
    copyType_ = getCopyType(dst, src);
    curStream_ = curStream;

    recomputeTensorsInfo(dst, src);
  }

  void updateCopyType(DIPUCopyType copyType) { copyType_ = copyType; }
};

inline void doSrcStreamWaitDstStream(const CopyParamsInfo& info,
                                     bool block_cpu) {
  DIPUGuard dstGuard(info.dstDevice_);
  DIPUEvent dstEvent;
  dstEvent.record(dipu::getCurrentDIPUStream(info.dstDevice_));
  dstGuard.set_index(info.srcDevice_);
  if (block_cpu) {
    dstEvent.synchronize();
  } else {
    dstEvent.wait(info.curStream_);
  }
}

inline void doDstStreamWaitSrcStream(const CopyParamsInfo& info,
                                     bool block_cpu) {
  DIPUEvent srcEvent;
  srcEvent.record(info.curStream_);
  DIPUGuard dstGuard(info.dstDevice_);
  if (block_cpu) {
    srcEvent.synchronize();
  } else {
    srcEvent.wait(dipu::getCurrentDIPUStream(info.dstDevice_));
  }
}

// process sync logic when copy between device and device
// 1. when on same device, non_blocking will always be True and no need sync
// stream.
// 2. when on difference device
//    2.1 non_blocking=False, sync stream.
//    2.2 non_blocking=True, recordStream to ensure tensor free
//    safety and doDstStreamWaitSrcStream to ensure copy complete before used.
inline void tryRecordOrSyncStreamD2D(const CopyParamsInfo& info,
                                     const at::Tensor& dst,
                                     const at::Tensor& src,
                                     DIPUStream& cur_stream, bool non_blocking,
                                     bool block_cpu) {
  if (!non_blocking) {
    cur_stream.synchronize();
    return;
  }

  // When copy from device to device, torch allocator ensure that malloc and
  // free are in the same stream, so it is not necessary to recordStream.
  if (!isTorchAllocator()) {
    const bool is_default_stream = (dipu::getDefaultDIPUStream() == cur_stream);
    if (!is_default_stream) {
      recordStream(dst, cur_stream);
      recordStream(src, cur_stream);
    }
  }
  if (info.copyType_ == DIPUCopyType::D2OtherD) {
    doDstStreamWaitSrcStream(info, block_cpu);
  }
}

// process sync logic when copy between host and device.
// 1. non_blocking=False, sync stream.
// 2. when cpu tensor is not pinned memory, sync stream when block_cpu=True.
// 3. when cpu tensor is pinned memory, record stream to ensure free safety.
inline void tryRecordOrSyncStreamHD(const CopyParamsInfo& info,
                                    const at::Tensor& dst,
                                    const at::Tensor& src,
                                    DIPUStream& cur_stream, bool non_blocking,
                                    bool block_cpu) {
  if (info.copyType_ == DIPUCopyType::H2H) {
    return;
  }
  if (!non_blocking) {
    cur_stream.synchronize();
    return;
  }

  const at::Tensor& cpu_tensor =
      (info.copyType_ == DIPUCopyType::H2D ? src : dst);
  const at::Tensor& device_tensor =
      (info.copyType_ == DIPUCopyType::H2D ? dst : src);
  bool is_pinned = isPinnedPtr(cpu_tensor.storage().data());
  // When copy between cpu tensor(not pinned) and device tensor, do sync stream
  // to ensure free safety while block_cpu is True.
  if (!is_pinned && block_cpu) {
    cur_stream.synchronize();
    return;
  }

  // copy between pin memory cpu tensor and device tensor
  if (!isTorchAllocator()) {
    if (is_pinned) {
      recordStream(cpu_tensor, cur_stream);
    }
    const bool is_default_stream = (dipu::getDefaultDIPUStream() == cur_stream);
    if (!is_default_stream) {
      recordStream(device_tensor, cur_stream);
    }
    return;
  }

  if (is_pinned) {
    TORCH_CHECK(allocator::CachingHostAllocator_recordEvent(
                    cpu_tensor.data_ptr(),
                    cpu_tensor.storage().data_ptr().get_context(), cur_stream),
                "CachingHostAllocator_recordEvent fail");
  }
}

inline void tryRecordOrSyncStream(const CopyParamsInfo& info,
                                  const at::Tensor& dst, const at::Tensor& src,
                                  DIPUStream& cur_stream, bool non_blocking,
                                  bool block_cpu_d2d, bool block_cpu_h2d) {
  bool is_all_device_tensor = (info.copyType_ == DIPUCopyType::D2Self ||
                               info.copyType_ == DIPUCopyType::D2OtherD);
  if (is_all_device_tensor) {
    tryRecordOrSyncStreamD2D(info, dst, src, cur_stream, non_blocking,
                             block_cpu_d2d);
    return;
  }

  tryRecordOrSyncStreamHD(info, dst, src, cur_stream, non_blocking,
                          block_cpu_h2d);
}

class DIPUCopyBase {
 public:
  DIPUCopyBase() = default;
  virtual ~DIPUCopyBase() = default;
  // throw(any type excep)
  virtual void run(at::Tensor& dst, const at::Tensor& src,
                   bool non_blocking) = 0;
};

/*
NOTICE: if input, output tensor occupy same storage size and has same
mem-format, DIPUCopyInplace will directly call mem_copy; if not, it call
copyNodirectXX.

DiopiCast: means call separate diopiCast func, it's a forward compatible
solutions because some vendor's DiopiCopy not support cast. new DiopiCopy api
require cast/
*/
template <bool DiopiCopy, bool DiopiCast>
class DIPUCopyInplace : public DIPUCopyBase {
 public:
  DIPUCopyInplace() = default;
  void run(at::Tensor& dst, const at::Tensor& src, bool non_blocking) override {
    TORCH_CHECK(dst.defined(), "dst is undefined");
    TORCH_CHECK(src.defined(), "src is undefined");
    if (dst.numel() == 0 || dst.is_same(src)) {
      return;
    }
    // We always perform the copy on the source device when do copy_d2d
    const DIPUGuard guard((!src.is_cpu()) ? src.device() : dst.device());
    auto curStream = dipu::getCurrentDIPUStream();

    auto info = CopyParamsInfo(dst, src, curStream);
    if (info.copyType_ == DIPUCopyType::D2Self) {
      non_blocking = true;
    }

    // Exit early if dst and src are views of the same data
    if ((dst.is_alias_of(src) && dst.storage_offset() == src.storage_offset() &&
         info.sameStride_ && info.sameDtype_)) {
      return;
    }
    checkOverlap(dst, src);
    if (native::dumpOpArgLevel() > 1) {
      std::cout << "	DIPUCopyInplace.run:	dst:" << native::dumpArg(dst)
                << std::endl;
      std::cout << "	DIPUCopyInplace.run::	src:" << native::dumpArg(src)
                << std::endl;
    }

    copyPreProcess(dst, src, non_blocking, info);

    copyAll(dst, src, non_blocking, info);

    copyPostProcess(dst, src, non_blocking, info, curStream);
  }

 protected:
  virtual void copyPreProcess(const at::Tensor& dst, const at::Tensor& src,
                              bool non_blocking, CopyParamsInfo& info) {
    if (DIPUCopyType::D2OtherD == info.copyType_) {
      doSrcStreamWaitDstStream(info, false);
    }
  }

  virtual void copyPostProcess(const at::Tensor& dst, const at::Tensor& src,
                               bool non_blocking, const CopyParamsInfo& info,
                               DIPUStream& curStream) {
    // 1. block_cpu_d2d=False default, because we do not need sync stream when
    // copy on two devices, just wait between stream.
    // 2. block_cpu_h2d=True default, We need sync stream if cpu tensor is not
    // pin memory to ensure tensor free safety.
    tryRecordOrSyncStream(info, dst, src, curStream, non_blocking,
                          /* block_cpu_d2d = */ false,
                          /* block_cpu_h2d = */ true);
  }

  /*
  NOTICE: the memory area of the dst tensor (contains hollow area actually not
  belong to tensor) will be totally overwrited by the same-size src mem area.
  support copy between 2 tensor with same stride and dtype, no-dense and
  overlapped tensors are also supported. but the 2 cannot both be view (will
  casue data outside mem area be overwrited).
  */
  void doDirectMemFill(at::Tensor& dst, const at::Tensor& src,
                       DIPUStream& curStream, DIPUCopyType copyType,
                       bool needMemCpSync = true) {
    if (dst.is_view() && src.is_view()) {
      TORCH_CHECK(false, "doDirectMemFill cannot support all view-view copy");
    }

    memCopy(dst, src, curStream, copyType, /*nonOverlappingAndDense=*/false,
            /*isSynchronousCopy=*/false);

    if (needMemCpSync) {
      dipu::devproxy::syncStream(curStream.rawstream());
    }
  }

  // support mem copy between 2 nonOverlappingAndDense tensor with same stride
  // and dtype. both 2 can be view.
  void doDirectMemCopy(at::Tensor& dst, const at::Tensor& src,
                       DIPUStream& curStream, DIPUCopyType copyType,
                       bool needMemCpSync = true) {
    memCopy(dst, src, curStream, copyType, /*nonOverlappingAndDense=*/true,
            /*isSynchronousCopy=*/false);

    if (needMemCpSync) {
      dipu::devproxy::syncStream(curStream.rawstream());
    }
  }

  at::Tensor makeSameStrideTensor(const at::Tensor& src, DIPUStream& curStream,
                                  at::Device newDevice,
                                  bool willBackfillSrc = false) {
    /*
      1. src.is_contiguous() & contiguous_nhwc is not suitable here, if size of
       dim0 =1 and other dims are contiguous. it return true No matter what
       value the stride of dim0 is but create a new tensor with stride of dim0
       re-computed as contiguous.
      eg: src size [1, 3, 2, 2], with stride [any_value < 12, 1, 6, 3] is
       considered as channel last contiguous. and create new Tensor sameAsSrc
       has stride [12, 1, 6, 3].

      2. another problem: tensor.suggest_memory_format() is inconsistent with
       tensor.is_contiguous(c10::MemoryFormat::ChannelsLast) in above case.
       because the TensorImpl.is_channels_last_ and is_channels_last_contiguous_
       has differenet logic.
    */

    // if (src.is_contiguous(c10::MemoryFormat::ChannelsLast) ||
    //     src.is_contiguous()) {
    //   auto sameAsSrc =
    //       at::empty(src.sizes(), src.options().device(newDevice),
    //                 src.is_contiguous() ? c10::MemoryFormat::Contiguous
    //                                     : c10::MemoryFormat::ChannelsLast);
    //   return sameAsSrc;
    // }

    // empty_strided is much expensive than empty_memory_format().
    // see src/ATen/EmptyTensor.cpp computeStorageNbytes()
    auto sameAsSrc = at::empty_strided(src.sizes(), src.strides(),
                                       src.options().device(newDevice));
    // prefill newTensor to support backfill in future.
    if (willBackfillSrc && !src.is_non_overlapping_and_dense()) {
      doDirectMemFill(sameAsSrc, src, curStream, getCopyType(sameAsSrc, src));
    }
    return sameAsSrc;
  }

  // NOTICE: this func maximize leverage device copy (D2Self)
  // as relay in d2h, h2d, d2d copy. cannot used in device copy(D2Self).
  void doDeviceRelayCopy(at::Tensor& dst, const at::Tensor& src,
                         bool non_blocking, CopyParamsInfo& info) {
    switch (info.copyType_) {
      // create dst_device (relay, same stride)
      // 1. direct dst_cpu/otherdevice -> dst_device (is view).
      // 2. src_device -> dst_device. 3. direct dst_device ->
      // dst_cpu/otherdevice.
      case DIPUCopyType::D2OtherD:
      case DIPUCopyType::D2H: {
        auto curCopyType = info.copyType_;
        // same stride as dst.
        // TODO(fandaoyi):: check if D2OtherD need change device guard.
        auto dstInDevSrc =
            makeSameStrideTensor(dst, info.curStream_, src.device(), true);
        info.updateCopyType(DIPUCopyType::D2Self);
        copyNodirectOnDevice(dstInDevSrc, src, non_blocking, info);
        doDirectMemFill(dst, dstInDevSrc, info.curStream_, curCopyType, true);
      } break;
      // create src_device (relay, same stride)
      // direct src_cpu -> src_device, src_device -> dst(device)
      case DIPUCopyType::H2D: {
        auto srcInDstdev =
            makeSameStrideTensor(src, info.curStream_, dst.device());
        doDirectMemFill(srcInDstdev, src, info.curStream_, DIPUCopyType::H2D);
        info.updateCopyType(DIPUCopyType::D2Self);
        copyNodirectOnDevice(dst, srcInDstdev, non_blocking, info);
      } break;
      default:
        TORCH_CHECK(false,
                    "doDeviceRelayCopy not support one device device, it's a "
                    "proxy method");
    }
  }

  // NOTICE: doDeviceRelayCopy need create a relay tensor having same stride as
  // the dst/src. it's expensive if the tensor is a view with big hollow, so
  // supply this simple wrap method to help d2h, h2d, d2d copy. cannot used in
  // device copy(D2Self). logical approach:
  // 1. create dst_contig. 2. create src_contig and src -> src_contig.
  // 3. direct src_contig -> dst_contig  4. dst_contig -> dst
  //  (TODO(fandaoyi): automatic use)
  void doContigTensorRelayCopy(at::Tensor& dst, const at::Tensor& src,
                               bool non_blocking, CopyParamsInfo& info) {
    switch (info.copyType_) {
      case DIPUCopyType::D2OtherD:
      case DIPUCopyType::D2H: {
        // 1. create dst_contig. same device.
        auto dstContig =
            dst.is_contiguous()
                ? dst
                : at::empty_like(dst, c10::MemoryFormat::Contiguous);
        // TODO(fandaoyi): check if D2OtherD need change device guard.
        auto newInfo = CopyParamsInfo(dstContig, src, info.curStream_);
        if (newInfo.directMemCopy_) {
          doDirectMemCopy(dstContig, src, newInfo.curStream_,
                          newInfo.copyType_);
        } else {
          // equivalent as logical approach:
          // 2. create dst contig in src Device and do src -> dst_contigs_2.
          // 3. direct: dst_contigs_2(D) -> dst_contig(cpu/otherD).
          doDeviceRelayCopy(dstContig, src, non_blocking, newInfo);
        }
        // 4. dst_contig -> dst (in same device/cpu), this operation need
        // recurse call kernel, direcet copy cannot handle it.
        if (!dstContig.is_same(dst)) {
          dst.copy_(dstContig);
        }
      } break;
      case DIPUCopyType::H2D: {
        // 2. create src_contig and src -> src_contig (both cpu).
        auto srcContig = src.contiguous(c10::MemoryFormat::Contiguous);
        auto newInfo = CopyParamsInfo(dst, srcContig, info.curStream_);
        if (newInfo.directMemCopy_) {
          doDirectMemCopy(dst, srcContig, newInfo.curStream_,
                          newInfo.copyType_);
        }
        // equivalent as logical approach:
        // 1. create src_contig_2(D).  3. direct src_contig(CPU) -> src_contig_2
        // (D).
        // 4. src_contig_2 (device) -> dst (device),
        doDeviceRelayCopy(dst, srcContig, non_blocking, newInfo);
      } break;
      default:
        TORCH_CHECK(false,
                    "doDeviceRelayCopy not support one device device, it's a "
                    "proxy method");
    }
  }

  /*
  NOTICE:
  d2h: direct src (device) -> src_cpu. src_cpu -> dst (cpu)
  h2d: direct dst (device) -> dst_cpu (if view).. src (cpu)  -> dst_cpu.
       direct dst_cpu -> dst (device)
  d2d: direct src (device) -> src_cpu. direct dst (device) -> dst_cpu (if
  view). src_cpu -> dst_cpu.  direct dst_cpu -> dst (device), very very
  slow. this can handle any case, it's fallback solution.
 */
  void doCpuRelayCopy(at::Tensor& dst, const at::Tensor& src,
                      DIPUStream& curStream, bool non_blocking) {
    at::Tensor src_cpu = src;
    if (dipu::isDeviceTensor(src)) {
      src_cpu = makeSameStrideTensor(src, curStream,
                                     c10::Device(c10::DeviceType::CPU), false);
      // src storage size may bigger than src_cpu's when src is a partial view.
      // but not smaller. because src_cpu use same stride as src.
      // src -> src_cpu
      doDirectMemFill(src_cpu, src, curStream, DIPUCopyType::D2H);
    }

    if (dipu::isDeviceTensor(dst)) {
      auto dst_cpu = makeSameStrideTensor(
          dst, curStream, c10::Device(c10::DeviceType::CPU), true);
      // proxy to cpu to handle different type/view problem
      dst_cpu.copy_(src_cpu);
      // TODO(fandaoyi): ?? need further check ???
      // need force sync doDirectMemFill & slow down performance.
      // seems dipu CachedAllocator will recycle storage of temp tensor
      // when the tensor instance leave scope, even the stream(default stream)
      // not finish (not sure). is it a correct behavior ??
      // function doDirectMemCopy has same problem!
      doDirectMemFill(dst, dst_cpu, curStream, DIPUCopyType::H2D, true);
      return;
    }
    // dst is cpu
    dst.copy_(src_cpu);
  }

  // NOTICE: handle no-direct mem copy on one device, dipu has a simple
  // configurable template strategy which use DIOPI copy/cast correctly. if
  // vendor has no-complete implementation of DIOPI copy/cast. please override
  // copy_nodirect_device to decide the case needed to be executed by diopiCopy
  // and proxy other case back to 'doCpuRelayCopy' which contain a slow
  // implementaion.

  // 1. type cast. 2. expand/bcast. 3.1 special no-contiguous (stride
  // hollow/overlap) 3.2 mem-format no-contiguous (storage contiguous but not
  // nchw), we don't handle this
  virtual void copyNodirectOnDevice(at::Tensor& dst, const at::Tensor& src,
                                    bool non_blocking, CopyParamsInfo& info) {
    if (DiopiCast) {
      at::Tensor tmpSrc = src;
      if (!info.sameDtype_) {
        tmpSrc = native::dipu_wrap_diopi_cast_dtype(src, dst.scalar_type());
        info.recomputeTensorsInfo(dst, tmpSrc);
      }
      // after cast
      if (info.directMemCopy_) {
        doDirectMemCopy(dst, tmpSrc, info.curStream_, info.copyType_,
                        !tmpSrc.is_same(src));
      } else if (DiopiCopy) {
        native::dipu_wrap_diopi_copy_inp(dst, tmpSrc, non_blocking);
      } else {
        doCpuRelayCopy(dst, src, info.curStream_, non_blocking);
      }
    } else if (DiopiCopy) {  // !DiopiCast
      native::dipu_wrap_diopi_copy_inp(dst, src, non_blocking);
    } else {
      doCpuRelayCopy(dst, src, info.curStream_, non_blocking);
    }
  }

  // NOTICE: handle no-direct mem copy between different devices, dipu has
  // default strategy which use a intermidiate tensor, it's slow. vendor who has
  // more efficient p2p device copy can override it (eg: device has unified
  // addressing and supports passing in different device addresses to one kernel
  // can use copyNodirectOnDevice() to do 'between-device-copy')
  virtual void copyNodirectBetweenDevices(at::Tensor& dst,
                                          const at::Tensor& src,
                                          bool non_blocking,
                                          CopyParamsInfo& info) {
    if (DiopiCopy) {
      // doContigTensorRelayCopy(dst, src, non_blocking, info);
      doDeviceRelayCopy(dst, src, non_blocking, info);
      return;
    }
    // if diopiCopy = false, direct do cpu copy is best.
    doCpuRelayCopy(dst, src, info.curStream_, non_blocking);
  }

  // NOTICE: copy no-direct mem copy between cpu and device, dipu has default
  // strategy use intermidiate tensor, it's slow. vendor who has more efficient
  // solution can override it.
  virtual void copyNodirectDeviceHost(at::Tensor& dst, const at::Tensor& src,
                                      bool non_blocking, CopyParamsInfo& info) {
    if (DiopiCopy) {  // try to maximum leverage device copy,
      // doContigTensorRelayCopy(dst, src, non_blocking, info);
      doDeviceRelayCopy(dst, src, non_blocking, info);
      return;
    }
    // if diopiCopy = false, direct do cpu copy is best.
    doCpuRelayCopy(dst, src, info.curStream_, non_blocking);
  }

  // This virtual method, which is simply a wrapper of doDirectMemCopy by
  // default, is only used in copyAll. It keeps all original information
  // including non_blocking, and is thus suitable for overriding on different
  // devices for more control of the direct memory copy process
  virtual void directMemCopy(at::Tensor& dst, const at::Tensor& src,
                             CopyParamsInfo& info, bool non_blocking) {
    doDirectMemCopy(dst, src, info.curStream_, info.copyType_,
                    /*needMemCpSync=*/false);
  }

  // overriding this func is possible but not recommended
  virtual void copyAll(at::Tensor& dst, const at::Tensor& src,
                       bool non_blocking, CopyParamsInfo& info) {
    at::Tensor tmpSrc = src;
    if (!info.sameSize_) {
      tmpSrc = src.expand_as(dst);
      info.recomputeTensorsInfo(dst, tmpSrc);
    }
    if (info.directMemCopy_) {
      directMemCopy(dst, tmpSrc, info, non_blocking);
      return;
    }
    switch (info.copyType_) {
      case DIPUCopyType::D2Self:
        copyNodirectOnDevice(dst, tmpSrc, non_blocking, info);
        break;
      case DIPUCopyType::D2OtherD:
        copyNodirectBetweenDevices(dst, tmpSrc, non_blocking, info);
        break;
      default:
        copyNodirectDeviceHost(dst, tmpSrc, non_blocking, info);
    }
  }
};
using DIPUCopyInpOnCPU = DIPUCopyInplace<false, false>;
using DIPUCopyInpOnDIOPI = DIPUCopyInplace<true, false>;
using DIPUCopyInpOnDIOPIWithCast = DIPUCopyInplace<true, true>;

DIPUCopyBase* getDipuCopyInstance();

void setDipuCopyInstance(DIPUCopyBase* op);

}  // namespace dipu
