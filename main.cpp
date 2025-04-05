#include <boost/lockfree/queue.hpp>
#include <vector>
#include <thread>
#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <unistd.h>

#define BATCH_SIZE 64
#define PAYLOAD_SIZE 1024

// Global lock-free queue
boost::lockfree::queue<std::vector<uint8_t>*> packetQueue(1024);

// Thread pinning
void setThreadAffinity(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        perror("Failed to set thread affinity");
    }
}

// CUDA kernel
__global__ void aesEncryptKernel(uint8_t* input, uint8_t* output, size_t length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        output[idx] = input[idx] ^ 0xFF;
    }
}

// GPU encryption with error handling
bool encryptOnGPU(uint8_t* h_input, uint8_t* h_output, size_t size, cudaStream_t stream) {
    uint8_t* d_input = nullptr;
    uint8_t* d_output = nullptr;

    if (cudaMalloc(&d_input, size) != cudaSuccess ||
        cudaMalloc(&d_output, size) != cudaSuccess) {
        std::cerr << "CUDA malloc failed\n";
        return false;
    }

    if (cudaMemcpyAsync(d_input, h_input, size, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        std::cerr << "CUDA memcpy to device failed\n";
        return false;
    }

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    aesEncryptKernel<<<numBlocks, blockSize, 0, stream>>>(d_input, d_output, size);

    if (cudaGetLastError() != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed\n";
        return false;
    }

    if (cudaMemcpyAsync(h_output, d_output, size, cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        std::cerr << "CUDA memcpy to host failed\n";
        return false;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return true;
}

// Producer thread
void payloadProducer() {
    setThreadAffinity(0); // Core 0
    while (true) {
        auto* payload = new std::vector<uint8_t>(PAYLOAD_SIZE, 0xAB);
        while (!packetQueue.push(payload)) { usleep(1); }
    }
}

// Consumer thread
void payloadConsumer(int sockfd, struct sockaddr_in* dest_addr) {
    setThreadAffinity(1); // Core 1

    struct mmsghdr msgs[BATCH_SIZE];
    struct iovec iovecs[BATCH_SIZE];
    std::vector<uint8_t>* payloads[BATCH_SIZE];

    uint8_t* pinned_input;
    uint8_t* pinned_output;

    if (cudaHostAlloc(&pinned_input, BATCH_SIZE * PAYLOAD_SIZE, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(&pinned_output, BATCH_SIZE * PAYLOAD_SIZE, cudaHostAllocDefault) != cudaSuccess) {
        std::cerr << "Pinned memory allocation failed\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    while (true) {
        for (int i = 0; i < BATCH_SIZE; ++i) {
            while (!packetQueue.pop(payloads[i])) { usleep(1); }
            std::memcpy(pinned_input + i * PAYLOAD_SIZE, payloads[i]->data(), PAYLOAD_SIZE);
            delete payloads[i];
        }

        if (!encryptOnGPU(pinned_input, pinned_output, BATCH_SIZE * PAYLOAD_SIZE, stream)) {
            std::cerr << "Encryption failed\n";
            continue;
        }

        cudaStreamSynchronize(stream);

        for (int i = 0; i < BATCH_SIZE; ++i) {
            iovecs[i].iov_base = pinned_output + i * PAYLOAD_SIZE;
            iovecs[i].iov_len = PAYLOAD_SIZE;
            msgs[i].msg_hdr.msg_iov = &iovecs[i];
            msgs[i].msg_hdr.msg_iovlen = 1;
            msgs[i].msg_hdr.msg_name = dest_addr;
            msgs[i].msg_hdr.msg_namelen = sizeof(*dest_addr);
        }

        if (sendmmsg(sockfd, msgs, BATCH_SIZE, 0) < 0) {
            perror("sendmmsg failed");
        }
    }

    cudaFreeHost(pinned_input);
    cudaFreeHost(pinned_output);
    cudaStreamDestroy(stream);
}

// Main
int main() {
    int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (sockfd < 0) {
        perror("Socket creation failed");
        return 1;
    }

    struct sockaddr_in dest_addr{};
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(12345);
    if (inet_pton(AF_INET, "192.168.1.1", &dest_addr.sin_addr) != 1) {
        perror("Invalid IP");
        return 1;
    }

    std::thread producer(payloadProducer);
    std::thread consumer(payloadConsumer, sockfd, &dest_addr);

    producer.join();
    consumer.join();

    return 0;
}
