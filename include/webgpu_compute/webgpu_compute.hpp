#pragma once

#include <vector>
#include <webgpu/webgpu.h>
#include <iostream>
#include <string>

class WebGPUCompute {
public:
    WebGPUCompute();
    std::vector<float> perform_aggregation(std::vector<std::vector<float>> data);

private:
    void webgpu_vector_addition(std::vector<float>& a, std::vector<float>& b, std::vector<float>& result);
    void cleanup();
    
    void create_bind_group();
    void initialize_pipeline();
    void create_buffers(std::vector<float>& a, std::vector<float>& b);
    void initialize_device();

    const char* shaderCode = R"(
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read> b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> result: array<f32>;
        
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let index = global_id.x;
            if (index < arrayLength(&a)) {
                result[index] = a[index] + b[index];
            }
        }
    )";

    WGPUInstance instance;
    WGPUAdapter adapter;
    WGPUDevice device;
    WGPUShaderModule shaderModule;
    WGPUComputePipeline pipeline;
    WGPUBuffer bufferA;
    WGPUBuffer bufferB;
    WGPUBuffer bufferResult;
    WGPUBuffer stagingBuffer;
    WGPUBindGroup bindGroup;

    std::string shaderEntryPoint = "main";
};