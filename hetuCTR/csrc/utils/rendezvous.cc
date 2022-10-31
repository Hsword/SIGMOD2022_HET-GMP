#include "rendezvous.h"

using namespace hetuCTR;

TCPRendezvous::TCPRendezvous(int rank, int nrank, std::string ip, int port)
{
  CHECK(rank < nrank && nrank > 0 && rank >= 0);
  rank_ = rank;
  nrank_ = nrank;
  addr_ = "tcp://"+ ip + ":" + std::to_string(port);
  context_ = zmq_ctx_new();
  CHECK(context_ != NULL) << "create 0mq context failed";
  if (rank == 0) {
    bind();
  } else {
    connect();
  }
}

void TCPRendezvous::bind() {
  socket_ = zmq_socket(context_, ZMQ_ROUTER);
  CHECK(socket_ != NULL)
      << "create root socket failed: " << zmq_strerror(errno);
  int ret_val = zmq_bind(socket_, addr_.c_str());
  CHECK(ret_val == 0)
      << "Bind to " << addr_ << " failed : " << zmq_strerror(errno);
}

void TCPRendezvous::connect() {
  socket_ = zmq_socket(context_, ZMQ_DEALER);
  CHECK(socket_ != NULL)
      << "create worker socket failed: " << zmq_strerror(errno);
  std::string my_id = std::to_string(rank_);
  zmq_setsockopt(socket_, ZMQ_IDENTITY, my_id.data(), my_id.size());
  int ret_val = zmq_connect(socket_, addr_.c_str());
  CHECK(ret_val == 0)
      <<  "Connect to " + addr_ + " failed : " + zmq_strerror(errno);
}

static void FreeData(void *data, void *hint) {}

void TCPRendezvous::broadcast(void *data, size_t len) {
  if (rank_ == 0) {
    for (int i = 0; i < nrank_ - 1; i++) {
      zmq_msg_t msg, head, body;
      zmq_msg_init(&head);
      zmq_msg_init(&body);
      zmq_msg_init_data(&msg, data, len, FreeData, NULL);

      zmq_msg_recv(&head, socket_, 0);
      zmq_msg_recv(&body, socket_, 0);
      zmq_msg_send(&head, socket_, ZMQ_SNDMORE);
      zmq_msg_send(&msg, socket_, 0);

      zmq_msg_close(&head);
      zmq_msg_close(&body);
      zmq_msg_close(&msg);
    }
  } else {
    char buf[1];
    zmq_send(socket_, buf, 1, 0);
    zmq_recv(socket_, data, len, 0);
  }
}

TCPRendezvous::~TCPRendezvous() {
  // close sockets
  if (socket_) {
    int linger = 0;
    int rc = zmq_setsockopt(socket_, ZMQ_LINGER, &linger, sizeof(linger));
    CHECK(rc == 0 || errno == ETERM);
    CHECK_EQ(zmq_close(socket_), 0);
    socket_ = nullptr;
  }
  // close 0mq context
  if (context_) {
    zmq_ctx_destroy(context_);
    context_ = nullptr;
  }
}
