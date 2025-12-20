#include "webgpu_compute/webgpu_compute.hpp"

WebGPUCompute::WebGPUCompute() {
}

void WebGPUCompute::initialize_device() {
    // 1. Create WebGPU instance and adapter
    WGPUInstanceDescriptor instanceDesc = {};
    this->instance = wgpuCreateInstance(&instanceDesc);
    
    WGPURequestAdapterOptions adapterOpts = {};
    
    auto onAdapterRequestEnded = [](WGPURequestAdapterStatus status, WGPUAdapter adapter, char const* message, void* userdata) {
        if (status == WGPURequestAdapterStatus_Success) {
            *static_cast<WGPUAdapter*>(userdata) = adapter;
        }
    };

    wgpuInstanceRequestAdapter(this->instance, &adapterOpts, onAdapterRequestEnded, &this->adapter);
    
    WGPUDeviceDescriptor deviceDesc = {};
    
    auto onDeviceRequestEnded = [](WGPURequestDeviceStatus status, WGPUDevice device, char const* message, void* userdata) {
        if (status == WGPURequestDeviceStatus_Success) {
            *static_cast<WGPUDevice*>(userdata) = device;
        }
    };
    wgpuAdapterRequestDevice(this->adapter, &deviceDesc, onDeviceRequestEnded, &this->device);
}

void WebGPUCompute::create_buffers(std::vector<float>& a, std::vector<float>& b) {
    size_t bufferSize = size * sizeof(float);
    
    WGPUBufferDescriptor bufferDescA = {};
    bufferDescA.size = bufferSize;
    bufferDescA.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    this->bufferA = wgpuDeviceCreateBuffer(this->device, &bufferDescA);
    
    WGPUBufferDescriptor bufferDescB = {};
    bufferDescB.size = bufferSize;
    bufferDescB.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    this->bufferB = wgpuDeviceCreateBuffer(this->device, &bufferDescB);
    
    WGPUBufferDescriptor bufferDescResult = {};
    bufferDescResult.size = bufferSize;
    bufferDescResult.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc;
    this->bufferResult = wgpuDeviceCreateBuffer(this->device, &bufferDescResult);
    
    WGPUBufferDescriptor stagingBufferDesc = {};
    stagingBufferDesc.size = bufferSize;
    stagingBufferDesc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    this->stagingBuffer = wgpuDeviceCreateBuffer(this->device, &stagingBufferDesc);
    
    wgpuQueueWriteBuffer(wgpuDeviceGetQueue(this->device), this->bufferA, 0, a.data(), bufferSize);
    wgpuQueueWriteBuffer(wgpuDeviceGetQueue(this->device), this->bufferB, 0, b.data(), bufferSize);
}

void WebGPUCompute::initialize_pipeline() {
    WGPUShaderModuleWGSLDescriptor wgslDesc = {};
    wgslDesc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
    wgslDesc.code = shaderCode;
    
    WGPUShaderModuleDescriptor shaderDesc = {};
    shaderDesc.nextInChain = &wgslDesc.chain;
    this->shaderModule = wgpuDeviceCreateShaderModule(this->device, &shaderDesc);
    
    // 6. Create compute pipeline
    WGPUComputePipelineDescriptor pipelineDesc = {};
    pipelineDesc.compute.module = this->shaderModule;
    pipelineDesc.compute.entryPoint = this->shaderEntryPoint;
    this->pipeline = wgpuDeviceCreateComputePipeline(this->device, &pipelineDesc);
}

void WebGPUCompute::create_bind_group() {
    WGPUBindGroupEntry entries[3] = {};
    entries[0].binding = 0;
    entries[0].buffer = bufferA;
    entries[0].size = bufferSize;
    entries[1].binding = 1;
    entries[1].buffer = bufferB;
    entries[1].size = bufferSize;
    entries[2].binding = 2;
    entries[2].buffer = bufferResult;
    entries[2].size = bufferSize;
    
    WGPUBindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.layout = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    bindGroupDesc.entryCount = 3;
    bindGroupDesc.entries = entries;
    this->bindGroup = wgpuDeviceCreateBindGroup(device, &bindGroupDesc);
}

std::vector<float> WebGPUCompute::perform_aggregation(std::vector<std::vector<float>> data) {
    std::vector<float> result(data[0].size(), 0.0f);
    for (int i =0; i< data.size(); i++) {
        this->webgpu_vector_addition(result, data[i], result);
    }
    return result;
}

void WebGPUCompute::webgpu_vector_addition(std::vector<float>& a, std::vector<float>& b, std::vector<float>& result) {
    size_t size = a.size();
    result.resize(size);

    this->initialize_device();
    this->create_buffers(a, b);
    this->initialize_pipeline();
    this->create_bind_group();
    
    // 8. Create command encoder and compute pass
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(this->device, nullptr);
    WGPUComputePassEncoder computePass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);
    wgpuComputePassEncoderSetPipeline(computePass, this->pipeline);
    wgpuComputePassEncoderSetBindGroup(computePass, 0, this->bindGroup, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(computePass, (size + 63) / 64, 1, 1);
    wgpuComputePassEncoderEnd(computePass);
    
    // 9. Copy result to staging buffer
    wgpuCommandEncoderCopyBufferToBuffer(encoder, this->bufferResult, 0, this->stagingBuffer, 0, this->bufferSize);
    
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(wgpuDeviceGetQueue(this->device), 1, &commands);
    
    // 10. Read back results
    wgpuBufferMapAsync(this->stagingBuffer, WGPUMapMode_Read, 0, this->bufferSize, 
        [](WGPUBufferMapAsyncStatus status, void* userdata) {}, nullptr);
    wgpuDevicePoll(this->device, true, nullptr);
    
    const float* mappedData = static_cast<const float*>(wgpuBufferGetConstMappedRange(this->stagingBuffer, 0, this->bufferSize));
    std::copy(mappedData, mappedData + size, result.begin());
    wgpuBufferUnmap(this->stagingBuffer);

    this->cleanup();
}

void WebGPUCompute::cleanup() {
    wgpuBufferRelease(this->bufferA);
    wgpuBufferRelease(this->bufferB);
    wgpuBufferRelease(this->bufferResult);
    wgpuBufferRelease(this->stagingBuffer);
    wgpuBindGroupRelease(this->bindGroup);
    wgpuComputePipelineRelease(this->pipeline);
    wgpuShaderModuleRelease(this->shaderModule);
    wgpuDeviceRelease(this->device);
    wgpuAdapterRelease(this->adapter);
    wgpuInstanceRelease(this->instance);
}