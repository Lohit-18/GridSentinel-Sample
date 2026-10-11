"""A faithful-but-offline model of IEC 61850-8-1 GOOSE messages.

GOOSE (Generic Object Oriented Substation Event) is the publish/subscribe
protocol substations use to multicast protection and measurement data with
millisecond latency.  Each publisher sends a dataset of values wrapped in an
APDU carrying control fields: ``stNum`` (incremented when a value *changes*),
``sqNum`` (incremented on every retransmission), a timestamp ``t``, a
``timeAllowedToLive`` and so on.

This module implements the APDU as real ASN.1 BER tag-length-value structures,
so the encode/decode round-trip behaves like the wire format — but nothing is
ever transmitted on a real network.  It exists purely to generate and inspect
synthetic traffic for the detector.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# APDU / context tags (IEC 61850-8-1, §A.3)
_TAG_APDU = 0x61
_CTX = {
    "gocbRef": 0x80, "timeAllowedToLive": 0x81, "datSet": 0x82, "goID": 0x83,
    "t": 0x84, "stNum": 0x85, "sqNum": 0x86, "test": 0x87, "confRev": 0x88,
    "ndsCom": 0x89, "numDatSetEntries": 0x8A, "allData": 0xAB,
}
_DATA_FLOAT = 0x87        # MMS Data: floating-point
_DATA_BOOL = 0x83         # MMS Data: boolean


# --------------------------------------------------------------------------- #
# BER TLV helpers
# --------------------------------------------------------------------------- #
def _enc_len(n: int) -> bytes:
    if n < 0x80:
        return bytes([n])
    out = []
    while n:
        out.append(n & 0xFF)
        n >>= 8
    out.reverse()
    return bytes([0x80 | len(out)]) + bytes(out)


def _tlv(tag: int, value: bytes) -> bytes:
    return bytes([tag]) + _enc_len(len(value)) + value


def _enc_uint(v: int) -> bytes:
    if v == 0:
        return b"\x00"
    out = []
    while v:
        out.append(v & 0xFF)
        v >>= 8
    out.reverse()
    return bytes(out)


def _enc_float(v: float) -> bytes:
    # IEC 61850 FloatingPoint: 1-byte exponent width (8) + IEEE-754 single
    return b"\x08" + struct.pack(">f", float(v))


def _read_tlv(buf: bytes, i: int) -> tuple[int, bytes, int]:
    """Return (tag, value_bytes, next_index)."""
    tag = buf[i]
    i += 1
    ln = buf[i]
    i += 1
    if ln & 0x80:
        nbytes = ln & 0x7F
        ln = int.from_bytes(buf[i:i + nbytes], "big")
        i += nbytes
    return tag, buf[i:i + ln], i + ln


def _dec_uint(b: bytes) -> int:
    return int.from_bytes(b, "big") if b else 0


# --------------------------------------------------------------------------- #
# Message model
# --------------------------------------------------------------------------- #
@dataclass
class GooseMessage:
    """One GOOSE APDU."""

    gocb_ref: str = "IED1/LLN0$GO$gcb01"
    time_allowed_to_live: int = 2000
    dat_set: str = "IED1/LLN0$Measurements"
    go_id: str = "GridSentinel/MU01"
    t: int = 0                                 # ms timestamp
    st_num: int = 1
    sq_num: int = 0
    test: bool = False
    conf_rev: int = 1
    nds_com: bool = False
    values: list[float] = field(default_factory=list)
    quality: list[bool] = field(default_factory=list)   # True = good quality
    appid: int = 0x3001

    @property
    def num_entries(self) -> int:
        return len(self.values) + len(self.quality)


def encode_goose(msg: GooseMessage) -> bytes:
    """Serialise a message to a BER-encoded GOOSE APDU (bytes)."""
    body = b""
    body += _tlv(_CTX["gocbRef"], msg.gocb_ref.encode())
    body += _tlv(_CTX["timeAllowedToLive"], _enc_uint(msg.time_allowed_to_live))
    body += _tlv(_CTX["datSet"], msg.dat_set.encode())
    body += _tlv(_CTX["goID"], msg.go_id.encode())
    body += _tlv(_CTX["t"], struct.pack(">Q", msg.t))
    body += _tlv(_CTX["stNum"], _enc_uint(msg.st_num))
    body += _tlv(_CTX["sqNum"], _enc_uint(msg.sq_num))
    body += _tlv(_CTX["test"], b"\x01" if msg.test else b"\x00")
    body += _tlv(_CTX["confRev"], _enc_uint(msg.conf_rev))
    body += _tlv(_CTX["ndsCom"], b"\x01" if msg.nds_com else b"\x00")
    body += _tlv(_CTX["numDatSetEntries"], _enc_uint(msg.num_entries))

    all_data = b""
    for v in msg.values:
        all_data += _tlv(_DATA_FLOAT, _enc_float(v))
    for q in msg.quality:
        all_data += _tlv(_DATA_BOOL, b"\x01" if q else b"\x00")
    body += _tlv(_CTX["allData"], all_data)

    return _tlv(_TAG_APDU, body)


def decode_goose(buf: bytes) -> GooseMessage:
    """Parse a BER-encoded GOOSE APDU back into a :class:`GooseMessage`."""
    tag, body, _ = _read_tlv(buf, 0)
    if tag != _TAG_APDU:
        raise ValueError(f"not a GOOSE APDU (tag=0x{tag:02x})")
    msg = GooseMessage(values=[], quality=[])
    i = 0
    while i < len(body):
        t, val, i = _read_tlv(body, i)
        if t == _CTX["gocbRef"]:
            msg.gocb_ref = val.decode(errors="replace")
        elif t == _CTX["timeAllowedToLive"]:
            msg.time_allowed_to_live = _dec_uint(val)
        elif t == _CTX["datSet"]:
            msg.dat_set = val.decode(errors="replace")
        elif t == _CTX["goID"]:
            msg.go_id = val.decode(errors="replace")
        elif t == _CTX["t"]:
            msg.t = struct.unpack(">Q", val)[0] if len(val) == 8 else _dec_uint(val)
        elif t == _CTX["stNum"]:
            msg.st_num = _dec_uint(val)
        elif t == _CTX["sqNum"]:
            msg.sq_num = _dec_uint(val)
        elif t == _CTX["test"]:
            msg.test = bool(val[0]) if val else False
        elif t == _CTX["confRev"]:
            msg.conf_rev = _dec_uint(val)
        elif t == _CTX["ndsCom"]:
            msg.nds_com = bool(val[0]) if val else False
        elif t == _CTX["numDatSetEntries"]:
            pass  # recomputed from parsed values
        elif t == _CTX["allData"]:
            j = 0
            while j < len(val):
                dt, dv, j = _read_tlv(val, j)
                if dt == _DATA_FLOAT and len(dv) == 5:
                    msg.values.append(struct.unpack(">f", dv[1:])[0])
                elif dt == _DATA_BOOL:
                    msg.quality.append(bool(dv[0]) if dv else False)
    return msg


# --------------------------------------------------------------------------- #
# Bridge between the grid measurement vector and GOOSE traffic
# --------------------------------------------------------------------------- #
def measurements_to_goose(z: np.ndarray, *, t_ms: int, st_num: int, sq_num: int,
                          go_id: str = "GridSentinel/MU01") -> GooseMessage:
    """Publish a measurement vector as one GOOSE dataset."""
    return GooseMessage(
        t=t_ms, st_num=st_num, sq_num=sq_num, go_id=go_id,
        values=[float(v) for v in z],
        quality=[True] * len(z),
    )


def goose_to_measurements(msg: GooseMessage) -> np.ndarray:
    """Recover the measurement vector from a GOOSE dataset."""
    return np.asarray(msg.values, dtype=float)


# --------------------------------------------------------------------------- #
# pcap export (so the traffic can be opened in Wireshark)
# --------------------------------------------------------------------------- #
_ETHERTYPE_GOOSE = 0x88B8


def _ethernet_goose_frame(apdu: bytes, appid: int) -> bytes:
    dst = bytes.fromhex("01 0c cd 01 00 01".replace(" ", ""))   # GOOSE multicast
    src = bytes.fromhex("00 11 22 33 44 55".replace(" ", ""))
    ethertype = struct.pack(">H", _ETHERTYPE_GOOSE)
    length = struct.pack(">H", len(apdu) + 8)
    goose_hdr = struct.pack(">H", appid) + length + b"\x00\x00\x00\x00"
    return dst + src + ethertype + goose_hdr + apdu


def write_pcap(path: str, messages: list[GooseMessage]) -> int:
    """Write GOOSE messages to a libpcap file. Returns bytes written."""
    out = struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)  # DLT_EN10MB
    for k, msg in enumerate(messages):
        frame = _ethernet_goose_frame(encode_goose(msg), msg.appid)
        ts_sec = msg.t // 1000
        ts_usec = (msg.t % 1000) * 1000
        out += struct.pack("<IIII", ts_sec, ts_usec, len(frame), len(frame))
        out += frame
    with open(path, "wb") as fh:
        fh.write(out)
    return len(out)
