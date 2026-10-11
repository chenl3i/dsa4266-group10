"""Nonlinear and temporal autoencoders for network attack detection on UGR'16 v2.

Models are fitted on background traffic only, because the training period has no
labelled attacks. Each is scored by reconstruction error and compared with a PCA
baseline at a threshold frozen on later training minutes (see docs/DSA4266.md).
"""

__version__ = "0.1.0"
