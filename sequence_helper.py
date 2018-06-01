import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq.python.ops.decoder import _transpose_batch_time
from tensorflow.python.framework import ops
from tensorflow.contrib.seq2seq.python.ops.helper import CustomHelper
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import array_ops



def _unstack_ta(inp):
    return tensor_array_ops.TensorArray(
        dtype=inp.dtype, size=array_ops.shape(inp)[0],
        element_shape=inp.get_shape()[1:]).unstack(inp)

class CustomInferenceHelper(CustomHelper):
    """ for the moment eos token must be -1 and pad token 0 so long as _zero_inputs are used in initialization.
     sequence_length als vektor mit einträgen übergeben. dabei muss der vektor als +1 gefeedet werden 
     (s. kommentar bei next_inputs_fn) """

    def __init__(self, inputs, sequence_length, n_outputs, n_inputs, eos_token=-1,
                 time_major=False, name=None):
        """Initializer.
        
        n_inputs und n_outputs aus shape abfragbar!
        """
        with ops.name_scope(name, "CustomInferenceHelper", [inputs, sequence_length,
                                                            eos_token, n_outputs, n_inputs]):

            self._sequence_length = tf.convert_to_tensor(sequence_length, name="sequence_length")
            if self._sequence_length.get_shape().ndims != 1:
                raise ValueError(
                    "Expected sequence_length to be a vector, but received shape: %s" %
                    self._sequence_length.get_shape())

            # bs needed as int for shape definition in next_inputs so that graph is defined as invariant
            self._batch_size = array_ops.size(sequence_length)
            # shape of decoder in feed for each step
            self._dec_depth = n_inputs + n_outputs

            # batch, for 1 step, n_feat + n_outputs of eos tokens in time major
            self._start_inputs = tf.ones([1, self._batch_size, self._dec_depth], dtype=tf.float32) * eos_token

            # batch, for 1 step, different required depths, zeros in time major
            self._zeros = tf.zeros([1, self._batch_size, self._dec_depth], dtype=tf.float32)
            self._zeros_inputs = tf.zeros([1, self._batch_size, n_inputs], dtype=tf.float32)
            self._zeros_outputs = tf.zeros([1, self._batch_size, n_outputs], dtype=tf.float32)

            # convert inputs to tensor and transpose batch time if necessary
            inputs = tf.convert_to_tensor(inputs, name="inputs")
            if not time_major:
                inputs = nest.map_structure(_transpose_batch_time, inputs)

            # concat concat_to inputs. serve no further function as will be skipped in next_inputs_fn. this way
            # conform with tf definition of next_inputs_fn and extraction of next_time from read_from_ta
            concat_inputs = nest.map_structure(lambda inp: tf.ones_like(inp), self._zeros_inputs)
            inputs = tf.concat([concat_inputs, inputs], axis=0)

            # applies _unstack_ta to inputs returning in the same inputs structure making them .read()-able
            # _zero_inputs returned directly as only one step, hence no .read and no _unstack_ta necessary
            self._input_tas = nest.map_structure(_unstack_ta, inputs)

        def initialize_fn(name=None):
            with ops.name_scope(name, "CustomInferenceHelperInitialize"):
                finished = math_ops.equal(0, self._sequence_length)
                all_finished = math_ops.reduce_all(finished)
                next_inputs = control_flow_ops.cond(
                    all_finished, lambda: self._zeros[0],
                    lambda: self._start_inputs[0])
                return (finished, next_inputs)

        def sample_fn(time, outputs, state, name=None):
            with ops.name_scope(name, "CustomInferenceHelperSample", [time, outputs]):
                del time, state
                sample_ids = tf.cast(tf.argmax(outputs, axis=-1), dtype=tf.int32)
                return sample_ids

        def next_inputs_fn(time, outputs, state, sample_ids, name=None):
            """next_inputs_fn for CustomInferenceHelper."""

            with ops.name_scope(name, "CustomInferenceHelperNextInputs",
                                [time, outputs, state]):
                next_time = time + 1
                finished = (next_time >= self._sequence_length)
                all_finished = math_ops.reduce_all(finished)

                def read_from_ta(inp):
                    return inp.read(next_time)

                next_inputs_prev_outputs = control_flow_ops.cond(all_finished,
                                                                 lambda: self._zeros_outputs[0],
                                                                 lambda: outputs)
                next_inputs_feat = control_flow_ops.cond(all_finished,
                                                         lambda: self._zeros_inputs[0],
                                                         lambda: nest.map_structure(read_from_ta, self._input_tas))
                next_inputs = tf.concat([next_inputs_feat, next_inputs_prev_outputs], axis=1)
                #next_inputs = tf.reshape(next_inputs, shape=[self._batch_size, self._dec_depth])
                return (finished, next_inputs, state)

        super().__init__(initialize_fn=initialize_fn, sample_fn=sample_fn, next_inputs_fn=next_inputs_fn)

class CustomTrainingHelper(CustomHelper):
    """A helper for use during training.  Only reads inputs.
    Returned sample_ids are the argmax of the RNN output logits. For regression tasks they do not serve any purpose.
    
    for the moment eos token must be -1 and pad token 0 so long as _zero_inputs are used in initialization.
    sequence_length als vektor mit einträgen übergeben. dabei muss der vektor als +1 gefeedet werden 
    (s. kommentar bei next_inputs_fn) """

    def __init__(self, inputs, sequence_length, n_outputs, n_inputs,
                 eos_token=-1, time_major=False, name=None):
        """Initializer.
        Args:
          inputs: A (structure of) input tensors. 
                  Contain [batch, sequence, features+labels (in this order as concatenation order in inf. helper!)]
          sequence_length: An int32 vector tensor.
          time_major: Python bool.  Whether the tensors in `inputs` are time major.
            If `False` (default), they are assumed to be batch major.
          name: Name scope for any created operations.
        Raises:
          ValueError: if `sequence_length` is not a 1D tensor.
        """

        with ops.name_scope(name, "CustomTrainingHelper", [inputs, sequence_length,
                                                           eos_token, n_outputs, n_inputs]):

            self._sequence_length = ops.convert_to_tensor(
                sequence_length, name="sequence_length")
            if self._sequence_length.get_shape().ndims != 1:
                raise ValueError(
                    "Expected sequence_length to be a vector, but received shape: %s" %
                    self._sequence_length.get_shape())

            batch_size = array_ops.size(sequence_length)

            # eos and pad tokens of length 1 step for entire batch
            start_inputs = tf.ones([batch_size, 1, n_inputs + n_outputs], dtype=tf.float32) * eos_token

            # transpose batch_time of eos and pad step
            start_inputs = nest.map_structure(_transpose_batch_time, start_inputs)

            inputs = ops.convert_to_tensor(inputs, name="inputs")

            if not time_major:
                inputs = nest.map_structure(_transpose_batch_time, inputs)

            # concat eos tokens to inputs after _transpose_batch_time
            inputs = tf.concat([start_inputs, inputs], axis=0, name="inputs")

            # other than inference helper no differentiation between inputs and outputs for padding sequence is needed
            # as 'inputs' already contains all features and labels and no concatenation over the two necessary
            self._zero_inputs = nest.map_structure(
                lambda inp: array_ops.zeros_like(inp[0, :]), inputs)

            self._input_tas = nest.map_structure(_unstack_ta, inputs)

        def initialize_fn(name=None):
            with ops.name_scope(name, "CustomTrainingHelperInitialize"):
                finished = math_ops.equal(0, self._sequence_length)
                all_finished = math_ops.reduce_all(finished)
                next_inputs = control_flow_ops.cond(
                    all_finished, lambda: self._zero_inputs,
                    lambda: nest.map_structure(lambda inp: inp.read(0), self._input_tas))
                return (finished, next_inputs)

        def sample_fn(time, outputs, name=None, **unused_kwargs):
            with ops.name_scope(name, "CustomTrainingHelperSample", [time, outputs]):
                sample_ids = tf.cast(tf.argmax(outputs, axis=-1), dtype=tf.int32)
                return sample_ids

        def next_inputs_fn(time, outputs, state, name=None, **unused_kwargs):
            """next_inputs_fn for CustomTrainingHelper."""

            # hängt dem standard training helper 1 hinterher, da dieser bereits die eos sequence gefeedet bekommt
            # implikationen für sequence length und max iterations im dynamic decoder?
            # um möglichen implikationen vorzubeugen hier sequence_length +1 (!) feeden, damit weit genug ausgerollt
            # wird, dabei jedoch weiterhin aus read(time) für aktuellen step readen
            with ops.name_scope(name, "CustomTrainingHelperNextInputs",
                                [time, outputs, state]):
                next_time = time + 1
                finished = (next_time >= self._sequence_length)
                all_finished = math_ops.reduce_all(finished)

                def read_from_ta(inp):
                    return inp.read(next_time)

                next_inputs = control_flow_ops.cond(
                    all_finished, lambda: self._zero_inputs,
                    lambda: nest.map_structure(read_from_ta, self._input_tas))
                return (finished, next_inputs, state)

        super().__init__(initialize_fn=initialize_fn, sample_fn=sample_fn, next_inputs_fn=next_inputs_fn)
