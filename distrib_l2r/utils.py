import pickle
import socket
import struct
from select import poll
from select import POLLIN
from typing import Any
from typing import Tuple
from typing import Union
import time


INT_SIZE = 4


def send_data(
    data: Any,
    addr: Tuple[str, Union[int, str]] = None,
    sock: socket.socket = None,
    reply: bool = False,
) -> Any:
    """Creates a TCP socket, optionally, and sends data to the specified address.
    If specified, listen for a response.

    :param data: any data that is either binary or able to be pickled
    :param addr: a tuple of (ip, port), if not provided, sock must not be none
    :param sock: a socket, if not provided, addr must not be none
    :param reply: listen on the same socket for a reply. if True, this
      function returns unpickled data
    """
    if not isinstance(data, bytes):
        data = pickle.dumps(data)

    if sock:
        send_bytes_with_prefix_size(msg=data, sock=sock)
        return wait_for_response(sock=sock) if reply else None

    else:
        if not addr:
            raise ValueError("send_data requires either a socket or an address")

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect(addr)
            send_bytes_with_prefix_size(msg=data, sock=sock)
            return wait_for_response(sock=sock) if reply else None


def wait_for_response(sock: socket.socket) -> Any:
    """Wait and return a response from a socket"""
    response = None
    polly = poll()
    polly.register(sock.fileno(), POLLIN)
    start_time = time.time()

    while not response:
        events = polly.poll(1)
        for fileno, event in events:
            if fileno == sock.fileno():
                return receive_data(sock=sock)
        total_time = time.time() - start_time
        if total_time > 10:
            print("WARNING: BLOCKING CALL IN SERVER.")


def receive_data(sock: socket.socket) -> Any:
    """Receive from a socket and unpickle"""
    return pickle.loads(recv_bytes_with_prefix_size(sock=sock))


def send_bytes_with_prefix_size(msg: bytes, sock: socket.socket) -> None:
    """Utility to send bytes across a socket"""
    if not isinstance(msg, bytes):
        raise TypeError

    # Prefix message with length and send
    sock.sendall(struct.pack(">I", len(msg)) + msg)


def recv_bytes_with_prefix_size(sock: socket.socket) -> bytes:
    """Utility to receive bytes that are prefixed with the size"""
    raw_size = sock.recv(INT_SIZE)
    msg_size = struct.unpack(">I", raw_size)[0]
    raw_data = b""

    # Continue receiving data until expected size is reached
    while len(raw_data) < msg_size:
        chunk = sock.recv(msg_size - len(raw_data))
        if not chunk:
            return None
        raw_data += chunk

    return raw_data
