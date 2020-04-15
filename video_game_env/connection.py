import socket
import subprocess
import platform
import time
import sys

import google.protobuf

from . import messages_pb2

class Connection():
    """Automatically starts the binary and creates a socket connection to it.

    When started with the default arguments, will start the binary on an open
    port and connect to it.

    If start_binary is set to False, the binary will
    not be automatically started, and connection will instead be made to the
    given address and port.

    Once connection has been made, the req member will be a protobuf Request
    class as defined in messages.proto. This member can be edited to set the
    message fields for the next request.

    The send_request() method will send the current request to the binary.
    """

    def __init__(self, address="localhost", port=None, start_binary=True, binary_path="main"):
        self.req = messages_pb2.Request()

        # Get a free port number
        if port is None:
            tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tcp.bind(("", 0))
            _, port = tcp.getsockname()
            tcp.close()

        # Start the binary
        if start_binary:
            try:
                if platform.system() == "Windows":
                    subprocess.Popen([binary_path, "-p", str(port)],
                                     stdout=subprocess.DEVNULL)
                else:
                    subprocess.Popen([binary_path, "-p", str(port)],
                                     stdout=subprocess.DEVNULL)
            except OSError:
                print("Starting the binary failed")
                sys.exit()

        # Attempt connecting until it succeeds
        for _ in range(10):
            try:
                self.s = socket.create_connection((address, port))
                break
            except ConnectionRefusedError:
                time.sleep(0.1)
                continue

        # Set TCP_NODELAY to prevent delays when sending short messages
        self.s.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)

    def send_request(self):
        """Send the Request message stored in this.req and resets
        it to default values.
        Returns the Response object received from the binary,
        or False if decoding the incoming message failed.
        """
        # Serialize message
        serialized = self.req.SerializeToString()

        # Reset the message to default values
        self.req = messages_pb2.Request()

        # Send message length
        msg_len = len(serialized)
        sent = self.s.send(msg_len.to_bytes(4, "big"))

        # Send message content
        total_sent = 0
        while total_sent < msg_len:
            sent = self.s.send(serialized[total_sent:])
            total_sent += sent

        # Receive message length
        data = b""
        while len(data) < 4:
            received = self.s.recv(4 - len(data))
            if len(received) == 0:
                raise ConnectionResetError("Connection was closed")
            data += received
        msg_len = int.from_bytes(data, "big")

        # Receive a Response message
        data = b""
        while len(data) < msg_len:
            received = self.s.recv(msg_len - len(data))
            if len(received) == 0:
                raise ConnectionResetError("Connection was closed")
            data += received

        # Try to parse the response and return it
        try:
            resp_msg = messages_pb2.Response()
            resp_msg.ParseFromString(data)
            return resp_msg

        # Return False if decoding fails
        except google.protobuf.message.DecodeError as e:
            print("DecodeError in reponse: {}".format(e))
            return False
