import sys
import os
path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(path, '../'))

# For XMem
sys.path.append("./third_party/XMem")
sys.path.append("./third_party/XMem/model")
sys.path.append("./third_party/XMem/util")
sys.path.append("./third_party/XMem/inference")

