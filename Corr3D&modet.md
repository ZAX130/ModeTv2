We use the modet package to implement the CUDA version of the 3D Correlation layer. Specifically, we simply set heads=1 and rpb=None, and scale=1/channel_num. For detailed code implementation, see Corr3D.py.

To run this layer:
1. Download and install the modet package. (cd modet; pip install .)
2. Download functional.py and Corr3D.py
