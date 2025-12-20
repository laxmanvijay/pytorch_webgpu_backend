#pragma once

#include <vector>
#include <string>
#include <sys/socket.h>
#include <vector>
#include <map>
// #include <fmt/core.h>
#include <numeric>
#include <netinet/tcp.h>
#include <netinet/ip.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
// #include <fmt/ranges.h>
#include <thread>
#include <algorithm>
#include <execution>
#include <iostream>


class TcpTransfer {
public:
    void perform_aggregation(std::vector<float> &data, int rank, int world_size);
    void send_to_switch(std::vector<float> &data, int rank, int world_size, int thread_id, int sock, sockaddr_in &server_addr);
    void receive_from_switch(std::vector<float> &data, int sock, int offset, int data_size, sockaddr_in &server_addr, int thread_offset);
};