"""GridSentinel — machine-learning defence for smart-grid substations.

GridSentinel detects *false data injection* (FDI) attacks against power-grid
state estimation and IEC 61850 GOOSE messaging. Everything runs on a fully
synthetic, self-contained simulation — no real substation, hardware or network
is ever touched.

Public surface
--------------
- ``gridsentinel.grid``       : IEEE test systems, DC power flow, WLS state estimation.
- ``gridsentinel.protocol``   : IEC 61850 GOOSE message model (encode / decode / pcap).
- ``gridsentinel.attacks``    : synthetic FDI + GOOSE attack generators (labelled data only).
- ``gridsentinel.features``   : feature extraction for the detectors.
- ``gridsentinel.detect``     : physics baseline + ML ensemble.
- ``gridsentinel.eval``       : experiments, metrics and comparisons.
- ``gridsentinel.report``     : self-contained HTML report.
"""
from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Lohit Shanmugam"
__license__ = "Apache-2.0"

__all__ = ["__version__", "__author__", "__license__"]
