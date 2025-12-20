#include "simulated_switch_server.h"


WebGPUTcpListener::WebGPUTcpListener(int port)
{
    // Create UDP socket
    sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock_fd < 0)
    {
        throw std::runtime_error("Failed to create socket");
    }

    // Configure server address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);

    // Bind socket
    if (bind(sock_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0)
    {
        throw std::runtime_error("Bind failed");
    }


    int flags = fcntl(sock_fd, F_GETFL, 0);
    fcntl(sock_fd, F_SETFL, flags | O_NONBLOCK);
}

ReceivedDataContainer &WebGPUTcpListener::store()
{
    return received_data_unquantized;
}

void WebGPUTcpListener::handle_packet()
{
    char buffer[1024];
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);

    fd_set read_fds;
    struct timeval tv;

    while (true)
    {
        FD_ZERO(&read_fds);
        FD_SET(sock_fd, &read_fds);

        // Check every 100ms
        tv.tv_sec = 0;
        tv.tv_usec = 100000;

        int ready = select(sock_fd + 1, &read_fds, NULL, NULL, &tv);

        if (ready < 0)
        {
            std::cerr << "Select error\n";
            continue;
        }

        if (FD_ISSET(sock_fd, &read_fds))
        {
            int bytes_received = recvfrom(sock_fd, buffer, sizeof(buffer), 0,
                                            (struct sockaddr *)&client_addr, &client_len);

            if (bytes_received < sizeof(PacketHeader))
            {
                std::cout << "Packet too small\n";
                continue;
            }

#ifdef DEBUG
            std::cout << "Received packet from " << inet_ntoa(client_addr.sin_addr) << "\n";
#endif

            // Unpack header
            PacketHeader *header = reinterpret_cast<PacketHeader *>(buffer);
            header->data_length = ntohl(header->data_length);
            header->rank = ntohl(header->rank);
            header->world_size = ntohl(header->world_size);
#ifdef DEBUG
            // print header
            std::cout << "Received packet with data length " << header->data_length << "\n";
            std::cout << "Received packet with rank " << header->rank << "\n";
            std::cout << "Received packet with world size " << header->world_size << "\n";
#endif
            this->current_world_size = header->world_size;
            this->current_received_size++;


            auto payload_unquantized = std::vector<float>(reinterpret_cast<float *>(buffer + sizeof(PacketHeader)), reinterpret_cast<float *>(buffer + sizeof(PacketHeader) + header->data_length * sizeof(float)));
            process_data<float_t>(header, payload_unquantized, client_addr);
            
        }
    }
}

void WebGPUTcpListener::process_data(PacketHeader *header, std::vector<float> &payload, const sockaddr_in &client_addr)
{
    
    store().add_data(payload, client_addr);

    #ifdef DEBUG
    std::cout << "Received data from rank " << header->rank << "\n";
    std::cout << "world_size: " << header->world_size << "\n";
    #endif

    // current_received_size maybe lower than world_size 
    // if we are receiving partial data
    if (store().get_size() == header->world_size)
    {
        #ifdef DEBUG
        std::cout << "Aggregating data of type " << header->quantization_type << "\n";
        #endif

        std::vector<T> result = aggregate_data<T>(store<T>().get_data());

        // store the size of the result at the beginning, 
        // this is used by the client to determine the size of the result in case of partial data
        result.insert(result.begin(), static_cast<T>(store().get_size() - this->dropped_packets));

        for (const auto &client : store().get_data())
        {
            sendto(sock_fd, result.data(), result.size() * sizeof(T), 0,
                    (struct sockaddr *)&client.second, sizeof(client.second));
        }

        this->reset();
        
        #ifdef DEBUG
        std::cout << "Sent result to all clients\n";
        #endif
    }
}

std::vector<float> WebGPUTcpListener::aggregate_data(const std::vector<std::pair<std::vector<float>, sockaddr_in>> &data)
{
    #ifdef DEBUG
    std::cout << "Aggregating " << data.size() << " data chunks\n";
    #endif

    std::vector<float> result(data[0].first.size());

    webgpu_compute.perform_aggregation(data, this->current_world_size);

    return result;
}

void WebGPUTcpListener::reset()
{
    this->received_data_quantized_8bit.clear();
    this->received_data_quantized_32bit.clear();
    this->received_data_unquantized.clear();

    this->previous_quantization_type = -1;
    this->current_world_size = 0;
    this->current_received_size = 0;
}

void WebGPUTcpListener::run()
{
    std::cout << "Server listening on port " << ntohs(server_addr.sin_port) << "\n";
    handle_packet();
}