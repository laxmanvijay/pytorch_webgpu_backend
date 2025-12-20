#pragma once
#include <vector>

class DataTransferManager {
public:
    void perform_aggregation_with_tcp(std::vector<float> data, int rank, int world_size);
    void perform_aggregation_with_rdma(std::vector<float> data, int rank, int world_size);
};