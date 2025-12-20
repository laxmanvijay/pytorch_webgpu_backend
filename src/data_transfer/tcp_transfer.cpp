#include "worker.h"


void TCPTranfer::perform_aggregation(std::vector<float> &data, int rank, int world_size)
{
    std::vector<std::thread> threads;

    int total_chunks = std::ceil((float)data.size() / (float)SIZE_OF_A_CHUNK);

    int num_threads = total_chunks < NUM_THREADS ? total_chunks : NUM_THREADS;

    for (int thread_id = 0; thread_id < num_threads; thread_id++)
    {
        threads.push_back(std::thread([&, thread_id]
                                        {
            int offset = thread_id;
            int curr_chunk = 0;

            int sock2 = socket(AF_INET, SOCK_DGRAM, 0);
            if (sock2 < 0) {
                throw std::runtime_error("Failed to create socket");
            }

            sockaddr_in server_addr2;
            
            memset(&server_addr2, 0, sizeof(server_addr2));
            server_addr2.sin_family = AF_INET;
            server_addr2.sin_addr.s_addr = inet_addr("127.0.0.1");
            server_addr2.sin_port = htons(SERVER_PORT + thread_id);

            while (true)
            {
                int jump_idx = offset + curr_chunk * num_threads;
                int start_idx = jump_idx * SIZE_OF_A_CHUNK;
                int end_idx = start_idx + SIZE_OF_A_CHUNK;

                if (start_idx >= data.size() || curr_chunk >= total_chunks)
                {
                    break;
                }
                
                // get the data chunk from jump_idx to jump_idx + chunk_size
                std::vector<float> chunk(data.begin() + start_idx, data.begin() + std::min(end_idx, (int)data.size()));

                send_to_switch<float>(chunk, rank, world_size, thread_id, sock2, server_addr2);

                receive_from_switch<float>(data, sock2, jump_idx, chunk.size(), server_addr2, offset);
                curr_chunk++;
            }

            close(sock2); }));
    }

    for (auto &thread : threads)
    {
        thread.join();
    }


    // Dequantize the data
    std::vector<float> result_buffer_dequantized;

    quantizer.dequantize(quantized_data, result_buffer_dequantized);

    // update the data inplace with the aggregated values

    for (int i = 0; i < data.size(); i++)
    {
        data[i] = result_buffer_dequantized[i];
    }

}

void TCPTranfer::send_to_switch(std::vector<float> &data, int rank, int world_size, int thread_id, int sock, sockaddr_in &server_addr)
{
    struct iovec iov[2];

    packet_info header;
    // Use standard network byte order functions
    header.rank = htonl(rank);
    header.world_size = htonl(world_size);
    header.data_size = htonl(data.size());

    iov[0].iov_base = &header;
    iov[0].iov_len = sizeof(header);

    iov[1].iov_base = data.data();

    iov[1].iov_len = data.size() * sizeof(float);

    // create an array of msghdr and add the iov array to it
    struct msghdr msg;

    memset(&msg, 0, sizeof(msg));
    msg.msg_name = &server_addr;
    msg.msg_namelen = sizeof(server_addr);
    msg.msg_iov = iov;
    msg.msg_iovlen = 2;

    if (sendmsg(sock, &msg, 0) < 0)
    {
        // fmt::println("Send failed: {}", strerror(errno));
        throw std::runtime_error("Send failed");
    }
}

void TCPTranfer::receive_from_switch(std::vector<float> &data, int sock, int offset, int data_size, sockaddr_in &server_addr, int thread_offset)
{
    // the switch always sends back in addition to the data, 
    // the size of the returned data at the beginning
    std::vector<float> result_buffer(data_size + 1);
    struct iovec iov[1];

    iov[0].iov_base = result_buffer.data();

    iov[0].iov_len = result_buffer.size() * sizeof(float);

    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_name = &server_addr;
    msg.msg_namelen = sizeof(server_addr);
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;

    int bytes_received = recvmsg(sock, &msg, 0);
    if (bytes_received < 0)
    {
        // fmt::println("Receive failed: {}", strerror(errno));
        throw std::runtime_error("Receive failed");
    }

    float data_size_received = result_buffer[0];

    int start_idx = offset * SIZE_OF_A_CHUNK;

    // skip the first element which is the size of the data
    std::vector<float> result_buffer_data;

    result_buffer_data.insert(result_buffer_data.end(), 
                        result_buffer.begin() + 1, 
                        result_buffer.end());

    // Check environment variable for averaging
    // divide the data by the number of workers participating in the aggregation.
    const char* skip_averaging = std::getenv(ENV_SKIP_AVERAGING);
    if (skip_averaging && std::string(skip_averaging) == "1") {
        #ifdef DEBUG
        // fmt::println("Averaging skipped");
        #endif
    }
    else {
        for (int i = 0; i < result_buffer_data.size(); i++) {
            result_buffer_data[i] /= result_buffer[0];
        }
    }

    for (int i = 0; i < result_buffer_data.size(); i++)
    {
        data[start_idx + i] = result_buffer_data[i];
    }
}