#pragma once

#include "common/logging.h"

#include <zmq.h>
#include <string>

namespace hetuCTR {

/**
 * @brief A helper class for Initializing NCCL connection
 *
 */
class TCPRendezvous
{
private:
  int rank_, nrank_;
  void *socket_ = nullptr;
  void *context_ = nullptr;
  std::string addr_;
  void bind();
  void connect();
public:
  /**
   * @brief Construct a new TCPRendezvous object
   *
   * @param rank rank of the worker
   * @param nrank total number of workers
   * @param ip IPv4 address
   * @param port IPv4 port
   */
  TCPRendezvous(int rank, int nrank, std::string ip, int port);
  /**
   * @brief broadcast data from rank 0 to other workers
   *
   * @param data points to the start of data
   * @param len length of the data
   */
  void broadcast(void *data, size_t len);
  /**
   * @brief Destroy the TCPRendezvous object
   *
   */
  ~TCPRendezvous();
};

} // namespace hetuCTR
