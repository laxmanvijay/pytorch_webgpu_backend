#include "webgpu_backend.hpp"

#define SERVER_PORT 30000

namespace c10d
{

  static WebGPUBackend* g_current_webgpu_backend = nullptr;

#ifdef IS_CUDA_BUILD
  void initializeStreamsEvents(
    const std::vector<at::Tensor>& tensors,
    std::vector<c10::Stream>& streams,
    std::vector<c10::Event>& events) {
    streams.reserve(tensors.size());
    events.reserve(tensors.size());
    for (const auto i : c10::irange(tensors.size())) {
      c10::Device device = tensors[i].device();
      c10::impl::VirtualGuardImpl impl(device.type());
      // Record event on current stream
      events.emplace_back(device.type());
      events[i].record(impl.getStream(device));
      // Get a non-default stream to execute asynchronous CUDA operations
      // on for this device. This ensures that the default stream used
      // by the caller is not occupied by c10d related operations.
      streams.push_back(
          impl.getStreamFromGlobalPool(device, /*isHighPriority=*/true));
      // Ensure the new stream is synchronized with the current stream.
      events[i].block(streams[i]);

      // `tensors` are created on a different stream. Hence, they must record
      // new streams in this Work to prevent being freed before the Work finishes.
      if (tensors[i].is_sparse()) {
        if (tensors[i].is_coalesced()) {
          impl.recordDataPtrOnStream(
              tensors[i].indices().storage().data_ptr(), streams[i]);
          impl.recordDataPtrOnStream(
              tensors[i].values().storage().data_ptr(), streams[i]);
        } else {
          // We will need to coalesce first, which means new tensors will
          // be allocated on the streams we just allocated, and there
          // is no need to record them separately.
        }
      } else {
        impl.recordDataPtrOnStream(tensors[i].storage().data_ptr(), streams[i]);
      }
    }
  }

  at::Tensor pinnedLike(at::Tensor& tensor) {
    auto* allocator = at::detail::getCUDAHooks().getPinnedMemoryAllocator();
    auto storage = c10::Storage(
        c10::Storage::use_byte_size_t(),
        static_cast<int64_t>(at::detail::computeStorageNbytes(
            tensor.sizes(), tensor.strides(), tensor.dtype().itemsize())),
        allocator,
        /*resizable=*/false);
    return at::empty({0}, tensor.options().device(at::kCPU))
        .set_(storage, 0, tensor.sizes(), tensor.strides());
  }
#endif

  WebGPUBackendWork::WebGPUBackendWork(OpType opType, std::vector<at::Tensor> &tensors, 
    int rank, int world_size, c10::intrusive_ptr<c10::ivalue::Future> future)
      : Work(-1, opType),
        tensors_(tensors),
        m_rank(rank),
        m_world_size(world_size),
        future_(future)
  {
    this->host_tensors_.reserve(this->tensors_.size());
#ifdef IS_CUDA_BUILD
    if (this->tensors_[0].is_cuda()) {
        initializeStreamsEvents(this->tensors_, this->streams_, this->events_);

        at::cuda::OptionalCUDAStreamGuard guard;
        guard.reset_stream(this->streams_[0]);
        this->host_tensors_.push_back(pinnedLike(this->tensors_[0]).copy_(this->tensors_[0], /*non_blocking=*/ true));
        this->on_cuda_ = true;
    } else {
        this->on_cuda_ = false;

        for (const auto i : c10::irange(this->tensors_.size())) {
          this->host_tensors_[i].copy_(this->tensors_[i], false);
        }
    }
#else
    this->on_cuda_ = false;
    for (const auto& tensor : this->tensors_) {
      this->host_tensors_.push_back(tensor.clone());
    }
#endif

    this->run();
  }

  bool WebGPUBackendWork::isCompleted() {
#ifdef IS_CUDA_BUILD
    if (!this->on_cuda_) {
      return true;
    }
    for (auto &event : this->events_) {
      if (!event.query()) {
        return false;
      }
    }
#endif
    return true;
  }

  bool WebGPUBackendWork::isSuccess() const {
#ifdef IS_CUDA_BUILD
    if (!this->on_cuda_) {
      return true;
    }
    for (auto &event : this->events_) {
      if (!event.query()) {
        return false;
      }
    }
#endif
    return true;
  }

  bool WebGPUBackendWork::wait(std::chrono::milliseconds) {
    return true;
  }

  c10::intrusive_ptr<c10::ivalue::Future> WebGPUBackendWork::getFuture() {
    return future_;
  }

  void WebGPUBackendWork::run() {
#ifdef IS_CUDA_BUILD
    if(this->on_cuda_) {
      // Synchronize with copy operations.
      for (auto &stream : this->streams_) {
        stream.synchronize();
      }
    }
#endif

    // After this point, the tensor data is on the CPU. (stored in host_tensors_)
    // We then perform the allreduce operation on the CPU tensors by first copying it to a combined buffer.

    size_t total_elements = 0;
    std::vector<size_t> tensor_sizes;

    // Calculate total size and store individual sizes
    for (const auto &tensor : this->host_tensors_)
    {
      tensor_sizes.push_back(tensor.numel());
      total_elements += tensor.numel();
    }

    // Allocate buffer for combined data
    std::vector<float> combined_data(total_elements);
    size_t offset = 0;

    // Copy all tensor data to combined buffer
    for (auto &tensor : this->host_tensors_)
    {
      auto t = tensor;
      if (!t.is_contiguous())
      {
      t = t.contiguous();
      }
      if (t.dtype() != torch::kFloat32)
      {
      t = t.to(torch::kFloat32);
      }

      size_t numel = t.numel();
      memcpy(combined_data.data() + offset,
         t.data_ptr<float>(),
         numel * sizeof(float));
      offset += numel;
    }

    // 1. Send tensors to webgpu for reduction
    WebGPUCompute::perform_aggregation(combined_data, this->m_rank, this->m_world_size);

    // Spread the combined data back to individual tensors
    offset = 0;

    for (size_t i = 0; i < this->host_tensors_.size(); i++)
    {
      auto &tensor = this->host_tensors_[i];
      auto numel = tensor_sizes[i];
      memcpy(tensor.data_ptr<float>(),
         combined_data.data() + offset,
         numel * sizeof(float));
      offset += numel;
    }

    // Copy the data back to the original tensors in the GPU.

#ifdef IS_CUDA_BUILD
    if (this->on_cuda_) {
      c10::OptionalStreamGuard guard;
      for (const auto i : c10::irange(this->tensors_.size())) {
        guard.reset_stream(streams_[i]);
        this->tensors_[i].copy_(this->host_tensors_[i], /* non_blocking */ true);
        events_[i].record(streams_[i]);
      }

      // this->synchronize();
    } else {
      // Copy the data back to the original tensors in the CPU.
      for (const auto i : c10::irange(this->tensors_.size())) {
        this->tensors_[i].copy_(this->host_tensors_[i], false);
      }
    }
#else
    // Copy the data back to the original tensors in the CPU.
    for (const auto i : c10::irange(this->tensors_.size())) {
      this->tensors_[i].copy_(this->host_tensors_[i], false);
    }
#endif
  }

  void IncBackendWork::synchronize() {
#ifdef IS_CUDA_BUILD
    if (!this->on_cuda_) {
      return;
    }
    // Synchronize with the copy back to CUDA tensors.
    for (const auto i : c10::irange(this->tensors_.size())) {
      c10::Device device = this->tensors_[i].device();
      events_[i].block(
          c10::impl::VirtualGuardImpl(device.type()).getStream(device));
    }
#endif
  }

  WebGPUBackend::WebGPUBackend(const c10::intrusive_ptr<::c10d::Store> &store,
            int rank,
            int size,
            const std::chrono::duration<float> &timeout,
            c10::intrusive_ptr<::c10d::ProcessGroupGloo::Options> options)
      : ProcessGroupGloo(store, rank, size, options),
        m_rank(rank),
        m_world_size(size)
  {
    g_current_inc_backend = this;
  }

  c10::intrusive_ptr<Work> WebGPUBackend::allreduce(
    std::vector<at::Tensor> &tensors,
    const AllreduceOptions &opts)
  {
    // 2. Create future to handle async completion
    auto future = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()));
    
    future->markCompleted(c10::IValue(tensors));

    return c10::make_intrusive<WebGPUBackendWork>(OpType::ALLREDUCE, tensors, this->m_rank, 
      this->m_world_size, std::move(future));
  }

  void WebGPUBackend::configure_backend(bool use_quantization,
      bool use_scaling, bool straggler_aware) {
    
        fmt::print("Configuring WebGPUBackend with quantization: {}, scaling: {}, straggler_aware: {}\n",
            use_quantization, use_scaling, straggler_aware);
  }

  c10::intrusive_ptr<Backend> WebGPUBackend::createWebGPUBackend(
      const c10::intrusive_ptr<::c10d::Store> &store,
      int rank,
      int size,
      const std::chrono::duration<float> &timeout)
  {
    auto options = c10d::ProcessGroupGloo::Options::create();
    options->devices.push_back(
        ::c10d::ProcessGroupGloo::createDeviceForHostname("127.0.0.1"));
    return c10::make_intrusive<IncBackend>(store, rank, size, timeout, options);
  }

  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
  {
    m.def("createWebGPUBackend", &WebGPUBackend::createWebGPUBackend);
    
    m.def("configure_backend", [](bool use_quantization, bool use_scaling, bool straggler_aware) {
        if (!g_current_webgpu_backend) {
            throw std::runtime_error("No WebGPUBackend instance found. Make sure you initialized with backend='webgpu_backend'.");
        }
        
        g_current_webgpu_backend->configure_backend(use_quantization, use_scaling, straggler_aware);
    },
    "Configure the WebGPUBackend with quantization, scaling, and straggler awareness options.",
    py::arg("use_quantization"), py::arg("use_scaling"), py::arg("straggler_aware"));
  }

}