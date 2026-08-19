import pytest
from colonyml.secure_communicator import SecureCommunicator


def make_comm(key="test-key"):
    return SecureCommunicator(
        local_ip="127.0.0.1",
        port=29500,
        secret_key=key
    )


def test_sign_and_verify():
    comm = make_comm()
    data = b"test gradient data"
    signed = comm._sign_message(data)
    verified = comm._verify_message(signed)
    assert verified == data


def test_tampered_message_rejected():
    comm = make_comm()
    data = b"test gradient data"
    signed = comm._sign_message(data)
    tampered = signed[:10] + b"X" * 22 + signed[32:]
    with pytest.raises(ValueError):
        comm._verify_message(tampered)


def test_wrong_key_rejected():
    sender = make_comm(key="correct-key")
    receiver = make_comm(key="wrong-key")
    data = b"secret gradient"
    signed = sender._sign_message(data)
    with pytest.raises(ValueError):
        receiver._verify_message(signed)


def test_empty_data_signed():
    comm = make_comm()
    data = b""
    signed = comm._sign_message(data)
    verified = comm._verify_message(signed)
    assert verified == data


def test_large_data_signed():
    import os
    comm = make_comm()
    data = os.urandom(1024 * 1024)  # 1MB
    signed = comm._sign_message(data)
    verified = comm._verify_message(signed)
    assert verified == data


def test_different_keys_incompatible():
    comm_a = make_comm(key="key-a")
    comm_b = make_comm(key="key-b")
    data = b"gradient"
    signed_a = comm_a._sign_message(data)
    signed_b = comm_b._sign_message(data)
    assert signed_a != signed_b