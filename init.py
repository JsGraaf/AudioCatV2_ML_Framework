import logging
import random

import numpy as np
import tensorflow as tf
import tensorrt as trt


def init(random_state: int = 42):

    logging.info(f"TF: {tf.__version__}")
    logging.info(f"GPUs: {tf.config.list_physical_devices('GPU')}")
    logging.info(f"TensorRT: {trt.__version__}")

    random.seed(random_state)
    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    # Tensorflow uniform random generator
    TF_G1 = tf.random.Generator.from_seed(random_state)
    tf.random.set_global_generator(TF_G1)

    tf.config.experimental.enable_op_determinism()
