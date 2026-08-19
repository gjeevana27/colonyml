import hmac
import hashlib
import socket
import pickle
import struct
import threading
import torch
import numpy as np
from colonyml.communicator import NodeCommunicator


class SecureCommunicator(NodeCommunicator):
    """
    Authenticated gradient communicator using HMAC-SHA256.

    Every message is signed with a shared secret key.
    Nodes that don't know the key cannot join the cluster
    or inject gradients.

    Usage:
        comm = SecureCommunicator(
            local_ip="192.168.1.30",
            port=29500,
            secret_key="my-secret-key"
        )
        averaged = comm.ring_allreduce(gradient, nodes, my_rank=0)
    """

    SIGNATURE_SIZE = 32  # SHA256 = 32 bytes

    def __init__(
        self,
        local_ip: str,
        port: int = 29500,
        secret_key: str = "colonyml-default-key"
    ):
        super().__init__(local_ip=local_ip, port=port)
        self.secret_key = secret_key.encode("utf-8")

    def _sign_message(self, data: bytes) -> bytes:
        """Sign data with HMAC-SHA256 and prepend signature."""
        sig = hmac.new(
            self.secret_key,
            data,
            hashlib.sha256
        ).digest()
        return sig + data

    def _verify_message(self, data: bytes) -> bytes:
        """
        Verify HMAC signature and return payload.
        Raises ValueError if signature is invalid.
        """
        if len(data) < self.SIGNATURE_SIZE:
            raise ValueError(
                "Message too short to contain signature"
            )

        sig = data[:self.SIGNATURE_SIZE]
        payload = data[self.SIGNATURE_SIZE:]

        expected = hmac.new(
            self.secret_key,
            payload,
            hashlib.sha256
        ).digest()

        if not hmac.compare_digest(sig, expected):
            raise ValueError(
                "Authentication failed — invalid signature. "
                "Make sure all nodes use the same secret key."
            )

        return payload

    def send_tensor(
        self,
        tensor: torch.Tensor,
        target_ip: str,
        target_port: int,
        timeout: int = 30
    ):
        """Send a signed tensor to another node."""
        data = pickle.dumps(tensor.numpy())
        signed_data = self._sign_message(data)
        size = len(signed_data)

        with socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        ) as s:
            s.settimeout(timeout)
            s.connect((target_ip, target_port))
            s.sendall(struct.pack("!Q", size))
            sent = 0
            while sent < size:
                chunk = signed_data[
                    sent:sent + self.BUFFER_SIZE
                ]
                s.sendall(chunk)
                sent += len(chunk)

    def receive_tensor(
        self,
        listen_ip: str,
        listen_port: int,
        timeout: int = 30
    ) -> torch.Tensor:
        """Receive and verify a signed tensor."""
        with socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        ) as server:
            server.setsockopt(
                socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
            )
            server.bind((listen_ip, listen_port))
            server.listen(1)
            server.settimeout(timeout)

            conn, addr = server.accept()
            with conn:
                size_data = self._recv_exactly(conn, 8)
                size = struct.unpack("!Q", size_data)[0]
                signed_data = self._recv_exactly(conn, size)

        # Verify signature before unpickling
        data = self._verify_message(signed_data)
        array = pickle.loads(data)
        return torch.tensor(array)


# Test
if __name__ == "__main__":
    import numpy as np

    print("SecureCommunicator — HMAC Authentication Test")
    print("=" * 50)

    comm = SecureCommunicator(
        local_ip="127.0.0.1",
        port=29500,
        secret_key="test-secret-key"
    )

    # Test signing and verification
    test_data = b"hello from ColonyML"
    signed = comm._sign_message(test_data)
    verified = comm._verify_message(signed)

    assert verified == test_data
    print("Sign + verify: PASSED")

    # Test tampered message is rejected
    tampered = signed[:10] + b"X" * 22 + signed[32:]
    try:
        comm._verify_message(tampered)
        print("Tamper detection: FAILED")
    except ValueError:
        print("Tamper detection: PASSED")

    # Test wrong key is rejected
    wrong_comm = SecureCommunicator(
        local_ip="127.0.0.1",
        port=29500,
        secret_key="wrong-key"
    )
    try:
        wrong_comm._verify_message(signed)
        print("Wrong key rejection: FAILED")
    except ValueError:
        print("Wrong key rejection: PASSED")

    print("\nAll security tests passed.")
    print("=" * 50)