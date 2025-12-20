#include "webgpu_compute.hpp"

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <any>
#include <fcntl.h>
#include <sys/select.h>
#include <cstring>

namespace IncComputeSimulatedSwitch
{
    struct PacketHeader
    {
        int32_t data_length;
        int32_t rank;
        int32_t world_size;
        int32_t offset;
        int32_t bit_width;
        int32_t quantization_type;
    };

    class ReceivedDataContainer
    {
        std::vector<std::pair<std::vector<float>, sockaddr_in>> received_data;

    public:
        void add_data(const std::vector<float> &data, const sockaddr_in &client_addr)
        {
            received_data.push_back({data, client_addr});
        }

        void clear()
        {
            received_data.clear();
        }

        int get_size()
        {
            return received_data.size();
        }

        std::vector<std::pair<std::vector<float>, sockaddr_in>> &get_data()
        {
            return received_data;
        }
    };

    class SimulatedSwitchServer: public TimerMixin, public RandomPacketDropMixin
    {
    private:
        int sock_fd;
        struct sockaddr_in server_addr;
        bool handle_struggler;
        int previous_quantization_type = -1;
        int current_world_size = 0;
        int current_received_size = 0;
        int dropped_packets = 0;

        ReceivedDataContainer received_data_unquantized;

        ReceivedDataContainer &store();

        WebGPUCompute webgpu_compute;

    public:
        SimulatedSwitchServer(int port, bool handle_struggler = false);

        void handle_packet();
        std::vector<float> aggregate_data(const std::vector<std::pair<std::vector<float>, sockaddr_in>> &data);
        void run();
        void reset();
    };
} // namespace IncComputeSimulatedSwitch