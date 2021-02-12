from numpy import *


# 改成flatten
def squeeze_vec(data_in):
    vec = array(data_in).flatten()
    return vec