#include "webgpu_tcp_listener.hpp"

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <port>\n";
            return 1;
        }

        int port = std::stoi(argv[1]);
        WebGPUTcpListener server(port);
        server.run();
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}