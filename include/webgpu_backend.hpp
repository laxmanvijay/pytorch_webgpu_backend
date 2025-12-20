#pragma once

#include <torch/extension.h>

#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#ifdef IS_CUDA_BUILD
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include <c10/util/irange.h>

#include <pybind11/chrono.h>

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <inccompute/worker.h>
#include <inccompute/quantizer/quantization_utils.h>
#include <netinet/tcp.h>
#include <netinet/ip.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

#define SIZE_OF_CHUNK 128
#define QUANTIZATION_SCALE 10000.0f
#define USE_CUDA_IF_AVAILABLE "USE_CUDA_IF_AVAILABLE"

namespace c10d
{

    class WebGPUBackend : public ProcessGroupGloo
    {
    public:
        int m_rank = -1;
        int m_world_size = -1;

        WebGPUBackend(const c10::intrusive_ptr<::c10d::Store> &store,
            int rank,
            int size,
            const std::chrono::duration<float> &timeout,
            c10::intrusive_ptr<Options> options = Options::create());

        c10::intrusive_ptr<Work> allreduce(
            std::vector<at::Tensor> &tensors,
            const AllreduceOptions &opts = AllreduceOptions()) override;

        c10::intrusive_ptr<Work> allreduce_with_quantization(
                std::vector<at::Tensor> &tensors,
                const AllreduceOptions &opts = AllreduceOptions());

        void configure_backend(bool use_quantization,
            bool use_scaling, bool straggler_aware);

        static c10::intrusive_ptr<Backend> createIncBackend(
            const c10::intrusive_ptr<::c10d::Store> &store,
            int rank,
            int size,
            const std::chrono::duration<float> &timeout);

        static void WebGPUBackendConstructor() __attribute__((constructor))
        {
            py::object module = py::module::import("torch.distributed");
            py::object register_backend =
                module.attr("Backend").attr("register_backend");
            register_backend("webgpu_backend", py::cpp_function(createWebGPUBackend), false, "cuda");
        }

        const std::string getBackendName() const override {
            return "webgpu_backend";
        }
    };

    class WebGPUBackendWork : public Work
    {
        friend class WebGPUBackend;

    public:
        WebGPUBackendWork(OpType opType, std::vector<at::Tensor> &tensors, 
            int rank, int world_size, c10::intrusive_ptr<c10::ivalue::Future> future);

        void run();
        bool isCompleted() override;
        bool isSuccess() const override;
        bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;
        virtual c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;
        void synchronize() override;

    private:
        c10::intrusive_ptr<c10::ivalue::Future> future_;
        std::vector<at::Tensor> tensors_;
        std::vector<at::Tensor> host_tensors_;
        std::vector<c10::Stream> streams_;
        std::vector<c10::Event> events_;
        bool on_cuda_;

        int m_rank;
        int m_world_size;
    };
}