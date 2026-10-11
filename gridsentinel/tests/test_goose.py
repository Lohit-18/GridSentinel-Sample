import os

import numpy as np

from gridsentinel.protocol.goose import (
    GooseMessage, encode_goose, decode_goose,
    measurements_to_goose, goose_to_measurements, write_pcap,
)


def test_encode_decode_roundtrip():
    msg = GooseMessage(st_num=9, sq_num=42, t=1700000000123,
                       values=[0.1, -2.3, 4.5], quality=[True, False, True])
    out = decode_goose(encode_goose(msg))
    assert out.st_num == 9 and out.sq_num == 42
    assert out.t == 1700000000123
    assert np.allclose(out.values, msg.values, atol=1e-5)
    assert out.quality == msg.quality


def test_apdu_tag_is_goose():
    buf = encode_goose(GooseMessage(values=[1.0]))
    assert buf[0] == 0x61                             # application-1 APDU tag


def test_measurement_bridge():
    z = np.array([0.5, -1.2, 3.3, 0.0])
    msg = measurements_to_goose(z, t_ms=1000, st_num=1, sq_num=0)
    assert np.allclose(goose_to_measurements(decode_goose(encode_goose(msg))), z, atol=1e-5)


def test_pcap_written(tmp_path):
    msgs = [measurements_to_goose(np.zeros(5), t_ms=i * 1000, st_num=1, sq_num=i)
            for i in range(10)]
    path = os.path.join(tmp_path, "t.pcap")
    n = write_pcap(path, msgs)
    assert n > 0 and os.path.getsize(path) == n
    with open(path, "rb") as fh:
        assert fh.read(4) == b"\xd4\xc3\xb2\xa1"      # little-endian pcap magic
