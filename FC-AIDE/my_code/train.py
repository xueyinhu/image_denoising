import tensorflow as tf

from my_code.config_global import parser_global
from my_code.nets import make_model

opt = parser_global
model = make_model(opt)

