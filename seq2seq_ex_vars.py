from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import RandomizedSearchCV
from sklearn.exceptions import NotFittedError

from datetime import datetime

import numpy as np

import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.layers.core import Dense
from tensorflow.python.layers.core import dense

import os
import inspect
import sys
current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(current_path)

from sequence_helper import CustomInferenceHelper, CustomTrainingHelper


class seq2seqModel(object):
    """ Class builds an instance of the seq2seq architecture as defined in meta_config. Meta_config contains 
    'meta_hyperparameters' that define the architecture and not the optimizer and regularization parameters. These are 
    fine tuned during another random search during the training session of the entire graph, as encoder and decoder are 
    trained separately as sub graphs. Optimizer and regularization parameters are part of config
    file sequence_helper.py must be saved to same directory as this file
    """

    def __init__(self, config):
        """ Initializer
        initializes seq2seqModel and calls builder functions for graph
        """

        self.config = config

        self._make_graph()

    def _make_graph(self):
        """ builds seq2seq graph according to self.config """

        self._init_placeholders()
        self._init_seq2seq()
        self._init_optimizer()

    def _init_placeholders(self):
        """ builds placeholders for graph """

        with tf.name_scope("inputs"):
            self.encoder_inputs = tf.placeholder(dtype=tf.float32,
                                                 shape=[None,
                                                        None,
                                                        self.config.n_inputs+self.config.n_outputs],
                                                 name="encoder_inputs")
            self.encoder_sequence_length = tf.placeholder(dtype=tf.int32,
                                                          shape=[None],
                                                          name="encoder_sequence_length")
            self.decoder_training_inputs = tf.placeholder(dtype=tf.float32,
                                                          shape=[None,
                                                                 None,
                                                                 self.config.n_inputs+self.config.n_outputs],
                                                          name="decoder_training_inputs")
            self.decoder_inference_inputs = tf.placeholder(dtype=tf.float32,
                                                          shape=[None, None, self.config.n_inputs],
                                                          name="decoder_inference_inputs")
            self.decoder_sequence_length = tf.placeholder(dtype=tf.int32,
                                                          shape=[None],
                                                          name="decoder_sequence_length")
            self.encoder_targets = tf.placeholder(dtype=tf.float32,
                                                  shape=[None, None, self.config.n_outputs],
                                                  name="encoder_targets")
            self.decoder_targets = tf.placeholder(dtype=tf.float32,
                                                  shape=[None, None, self.config.n_outputs],
                                                  name="decoder_targets")
            self.input_keep_prob = tf.placeholder_with_default(1.0, shape=())
            self.output_keep_prob = tf.placeholder_with_default(1.0, shape=())
            self.state_keep_prob = tf.placeholder_with_default(1.0, shape=())

    def rnn_cell(self, num_units):
        """ returns correct type of rnn_cell in accordance with self.config and self.meta_config 
        dropout wrapper does not yet work with layer norm lstm and is only wrapper around the other cells
        todo: - where to add activation selection from meta_config
              - where to add initializer from meta_config
              - convLSTM cells
              - activation function selection
              - further optimization parameters per cell if available
              - dropout wrapper
              - pyramid like shaped layers
        """

        with tf.name_scope("rnn_cell"):
            if self.config.cell_type == "lstm":
                cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units)
            elif self.config.cell_type == "lstm-peephole":
                cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units, use_peephole=True)
            elif self.config.cell_type == "gru":
                cell = tf.contrib.rnn.GRUCell(num_units=num_units)
            elif self.config.cell_type == "lstm-layer-norm":
                keep_prob = tf.minimum(self.input_keep_prob, self.output_keep_prob)
                return tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=num_units,
                                                             dropout_keep_prob=keep_prob)
            elif self.config.cell_type == "lstm-grid":
                cell = tf.contrib.rnn.GridLSTMCell(num_units=num_units)
            elif self.config.cell_type == "lstm-time-freq":
                cell = tf.contrib.rnn.TimeFreqLSTMCell(num_units=num_units)
            else:
                raise ValueError("Invalid rnn_cell: {0!s}".format(self.config.cell_type))

            return tf.nn.rnn_cell.DropoutWrapper(cell,
                                                 input_keep_prob=self.input_keep_prob,
                                                 output_keep_prob=self.output_keep_prob,
                                                 state_keep_prob=self.state_keep_prob)

    def _init_seq2seq(self):
        """ Builds seq2seq model including encoder and decoder
        Stores encoder and decoder trainable variables in separate lists for separate trainings
        todo: - add dropout functionality to training helper for label inputs
              - rescale label inputs, reverse afterwards
        """
        with tf.name_scope("rnn_cell"):
            enc_cells = [self.rnn_cell(num_units=self.config.n_neurons) for _ in range(self.config.n_layers)]
            multilayer_enc_cells = tf.contrib.rnn.MultiRNNCell(cells=enc_cells)

            dec_cells = [self.rnn_cell(num_units=self.config.n_neurons) for _ in range(self.config.n_layers)]
            multilayer_dec_cells = tf.contrib.rnn.MultiRNNCell(cells=dec_cells)

        decoder_output_layer = Dense(units=self.config.n_outputs, name="decoder_output_layer")

        with tf.variable_scope("encoder", initializer=self.config.initializer):
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=multilayer_enc_cells,
                                                               inputs=self.encoder_inputs,
                                                               sequence_length=self.encoder_sequence_length,
                                                               dtype=tf.float32)

            stacked_rnn_outputs = tf.reshape(encoder_outputs, shape=[-1, self.config.n_neurons])
            encoder_output_layer = dense(stacked_rnn_outputs, self.config.n_outputs, name="encoder_output_layer")
            self.encoder_outputs = tf.reshape(encoder_output_layer,
                                              shape=[-1, self.config.encoder_length, self.config.n_outputs],
                                              name="encoder_outputs")
        with tf.name_scope("decoder"):
            with tf.variable_scope("decoder", initializer=self.config.initializer):
                training_helper = CustomTrainingHelper(inputs=self.decoder_training_inputs,
                                                       sequence_length=self.decoder_sequence_length,
                                                       n_inputs=self.config.n_inputs,
                                                       n_outputs=self.config.n_outputs,
                                                       eos_token=self.config.eos_token,
                                                       name="training_helper")

                training_decoder = seq2seq.BasicDecoder(cell=multilayer_dec_cells,
                                                        helper=training_helper,
                                                        initial_state=encoder_state,
                                                        output_layer=decoder_output_layer)

                self.training_decoder_outputs = seq2seq.dynamic_decode(decoder=training_decoder)[0][0]

            with tf.variable_scope("decoder", reuse=True, initializer=self.config.initializer):
                inference_helper = CustomInferenceHelper(inputs=self.decoder_inference_inputs,
                                                         sequence_length=self.decoder_sequence_length,
                                                         n_inputs=self.config.n_inputs,
                                                         n_outputs=self.config.n_outputs,
                                                         eos_token=self.config.eos_token,
                                                         name="inference_helper")

                inference_decoder = seq2seq.BasicDecoder(cell=multilayer_dec_cells,
                                                         helper=inference_helper,
                                                         initial_state=encoder_state,
                                                         output_layer=decoder_output_layer)

                self.inference_decoder_outputs = seq2seq.dynamic_decode(decoder=inference_decoder)[0][0]

        self.encoder_weights = multilayer_enc_cells.weights[0]
        self.encoder_bias = multilayer_enc_cells.weights[1]
        self.decoder_weights = multilayer_dec_cells.weights[0]
        self.decoder_bias = multilayer_dec_cells.weights[1]

    def _init_optimizer(self):
        """ builds optimizer according to self.config 
        selecting encoder oder decoder variables freezes part of the model within the same graph
        todo: - further eval metrics such as mae, nmae, rmse, etc.
        """
        with tf.name_scope("optimization"):
            with tf.name_scope("encoder_training"):
                self.global_step_encoder = tf.Variable(0, trainable=False, name="global_step")
                self.encoder_loss = tf.reduce_mean(tf.square(self.encoder_targets - self.encoder_outputs))
                encoder_optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
                encoder_gradients = encoder_optimizer.compute_gradients(self.encoder_loss,
                                                                        var_list=tf.trainable_variables("encoder"))
                encoder_capped_gradients = [(tf.clip_by_value(grad,
                                                              -self.config.gradient_clip,
                                                              self.config.gradient_clip), var)
                                            for grad, var in encoder_gradients if grad is not None]
                self.encoder_train_op = encoder_optimizer.apply_gradients(encoder_capped_gradients,
                                                                          global_step=self.global_step_encoder)

            with tf.name_scope("decoder_training"):
                self.global_step_decoder = tf.Variable(0, trainable=False, name="global_step")
                self.decoder_loss_tr = tf.reduce_mean(tf.square(self.decoder_targets - self.training_decoder_outputs))
                self.decoder_loss_inf = tf.reduce_mean(tf.square(self.decoder_targets - self.inference_decoder_outputs))
                decoder_optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
                decoder_gradients = decoder_optimizer.compute_gradients(self.decoder_loss_tr,
                                                                        var_list=tf.trainable_variables("decoder"))
                decoder_capped_gradients = [(tf.clip_by_value(grad,
                                                              -self.config.gradient_clip,
                                                              self.config.gradient_clip), var)
                                            for grad, var in decoder_gradients if grad is not None]
                self.decoder_train_op = decoder_optimizer.apply_gradients(decoder_capped_gradients,
                                                                          global_step=self.global_step_decoder)
                #  ALTERNATIVE without gradient clipping
                # self.decoder_train_op = decoder_optimizer.minimize(self.decoder_loss_tr,
                #                                                    var_list=self.decoder_variable_list)

        with tf.name_scope("evaluation"):
            r_one = tf.constant(1, dtype=tf.float32)

            ## encoder
            encoder_ssr = tf.reduce_sum(tf.square(tf.subtract(self.encoder_targets, self.encoder_outputs)))
            encoder_tss = tf.reduce_sum(tf.square(tf.subtract(self.encoder_targets,
                                                              tf.reduce_mean(self.encoder_targets))))
            self.encoder_r_squared = tf.subtract(r_one, tf.divide(encoder_ssr, encoder_tss), name="encoder_r_squared")

            # self.backscaled_encoder_targets = self.inv_transform(self.encoder_targets)
            # self.backscaled_encoder_outputs = self.inv_transform(self.encoder_outputs)
            #
            # backscaled_encoder_loss = tf.reduce_mean(tf.square(
            #     self.backscaled_encoder_outputs - self.backscaled_encoder_targets))
            # self.encoder_rmse = tf.sqrt(backscaled_encoder_loss, name="encoder_rmse")

            ## decoder
            decoder_ssr_tr = tf.reduce_sum(tf.square(tf.subtract(self.decoder_targets, self.training_decoder_outputs)))
            decoder_ssr_inf = tf.reduce_sum(tf.square(tf.subtract(self.decoder_targets, self.inference_decoder_outputs)))
            decoder_tss = tf.reduce_sum(tf.square(tf.subtract(self.decoder_targets,
                                                              tf.reduce_mean(self.decoder_targets))))
            self.decoder_r_squared_tr = tf.subtract(r_one, tf.divide(decoder_ssr_tr, decoder_tss),
                                                    name="decoder_r_squared_training")
            self.decoder_r_squared_inf = tf.subtract(r_one, tf.divide(decoder_ssr_inf, decoder_tss),
                                                     name="decoder_r_squared_inference")

            # self.backscaled_decoder_targets = self.inv_transform(self.decoder_targets)
            # self.backscaled_decoder_outputs_tr = self.inv_transform(self.training_decoder_outputs)
            # self.backscaled_decoder_outputs_inf = self.inv_transform(self.inference_decoder_outputs)
            #
            # backscaled_decoder_loss_tr = tf.reduce_mean(tf.square(
            #     self.backscaled_decoder_targets - self.backscaled_decoder_outputs_tr))
            # backscaled_decoder_loss_inf = tf.reduce_mean(tf.square(
            #     self.backscaled_decoder_targets - self.backscaled_decoder_outputs_inf))
            # self.decoder_rmse_tr = tf.sqrt(backscaled_decoder_loss_tr, name="decoder_rmse_tr")
            # self.decoder_rmse_inf = tf.sqrt(backscaled_decoder_loss_inf, name="decoder_rmse_inf")

    # def inv_transform(self, X):
    #     """ inverse transforms an ndarray of shape [sequences, steps, depth]
    #     for backscaling of targets and outputs, does not work as sk does not take tf.array as inputs
    #     -> move towards config_search method for now as active session there can call array.eval() """
    #
    #     # X = X.eval()
    #     sequences, steps, depth = X.shape
    #     X = tf.reshape(X, [-1, depth])
    #
    #     X_backscaled = self.config.lab_pipeline.inverse_transform(X)
    #     X_backscaled = tf.reshape(X_backscaled, [sequences, steps, depth])
    #
    #     return X_backscaled


class seq2seqDecoderConfig(BaseEstimator, ClassifierMixin):
    """
    
    """

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 train_data,
                 val_data,
                 config=None,
                 model_restore_params=None,
                 batch_size=30,
                 learning_rate=0.01,
                 gradient_clip=1,
                 input_keep_prob=0.8,
                 output_keep_prob=0.8,
                 state_keep_prob=0.8,
                 label_keep_prob=0.8,
                 logging=True,
                 base_path=None):

        """ Initializer:
        
        
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gradient_clip = gradient_clip
        self.input_keep_prob = input_keep_prob
        self.output_keep_prob = output_keep_prob
        self.state_keep_prob = state_keep_prob
        self.label_keep_prob = label_keep_prob

        self.base_path = base_path
        self.logging = logging

        if config:
            self.config = config
        if model_restore_params:
            self.model_restore_params = model_restore_params
        self.n_layers = config.n_layers
        self.n_neurons = config.n_neurons
        self.cell_type = config.cell_type
        self.activation_fn = config.activation_fn
        self.decoder_length = config.decoder_length
        self.encoder_length = config.encoder_length
        self.eos_token = config.eos_token
        self.initializer = config.initializer
        self.n_epochs = config.n_epochs
        self.max_checks_wo_progress = config.max_checks_wo_progress

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.train_data = train_data
        self.val_data = val_data
        self.n_iterations = self.train_data.get_iterations(batch_size=self.batch_size)

        if train_data.prep_pipeline_lab:
            self.lab_pipeline = train_data.lab_scaler

        self._session = None
        self._graph = None
        self.model = None

    def _close_session(self):
        """ closes current session when called """
        if self._session is not None:
            self._session.close()

    def _get_model_params(self):
        """ Get all variable values """
        with self._graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value for gvar, value in zip(gvars, self._session.run(gvars))}

    def _restore_model_params(self, model_params):
        """ Set all variables to the given values """
        gvar_names = list(model_params.keys())
        assign_ops = {gvar_name: self._graph.get_operation_by_name(gvar_name + "/Assign")
                      for gvar_name in gvar_names}
        init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
        feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
        self._session.run(assign_ops, feed_dict=feed_dict)

    def fit(self, X, y=None, model_restore_params=None):
        """
        load and build graph first according to self.meta_config
        then fit encoder using sklearn to find best hyperparameters in self.config (call fit_encoder)
        restore best encoder model to graph and then train decoder using same sklearn with hyperparameters in
            self.config call fit_decoder
        restore best model decoder model and save to disk
        
        todo:
            - add option to not perform random search on encoder training but only build and initialize 
              a graph using specified architecture for when loading an already trained graph (= weights and biases)
            - decoder shall only be trained if certain encoder accuracy is reached (0.80?), if not, continue encoder 
              training or move on to next hyperparameter combination
        """

        self._close_session()

        now = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        self.log_dir = "{}/model/run-{}".format(self.base_path, now)
        self.save_path = "{}/model/run-{}/model_ckpt.ckpt".format(self.base_path, now)

        print("model run-{}".format(now))

        self._graph = tf.Graph()
        with self._graph.as_default() as graph:

            # build model and summaries

            model = seq2seqModel(config=self)

            tf.summary.scalar("encoder_loss",
                              model.encoder_loss,
                              collections=["decoder_training", "decoder_validation"])
            tf.summary.scalar("encoder_r2",
                              model.encoder_r_squared,
                              collections=["decoder_training", "decoder_validation"])
            tf.summary.scalar("decoder_loss_training",
                              model.decoder_loss_tr,
                              collections=["decoder_training", "decoder_validation"])
            tf.summary.scalar("decoder_loss_inference",
                              model.decoder_loss_inf,
                              collections=["decoder_training", "decoder_validation"])
            tf.summary.histogram("decoder_weights",
                                 model.decoder_weights,
                                 collections=["decoder_validation"])
            tf.summary.histogram("decoder_bias",
                                 model.decoder_bias,
                                 collections=["decoder_validation"])
            s_decoder_train = tf.summary.merge_all("decoder_training")
            s_decoder_val = tf.summary.merge_all("decoder_validation")

            # initialize graph
            init = tf.global_variables_initializer()
            self._saver = tf.train.Saver()

        # early stopping
        checks_wo_progress = 0
        best_loss = np.infty
        best_params = None

        self._session = tf.Session(graph=graph)
        with self._session.as_default() as sess:
            init.run()
            if model_restore_params:
                self._restore_model_params(model_restore_params)
            if self.logging:
                file_writer = tf.summary.FileWriter(self.log_dir, graph=graph)

            train_fetches = {"train_op": model.decoder_train_op,
                             "global_step_decoder_train": model.global_step_decoder,
                             "decoder_r2_inf": model.decoder_r_squared_inf,
                             "s_decoder_train": s_decoder_train}
            val_fetches = {"global_step_decoder_train": model.global_step_decoder,
                           "decoder_val_loss": model.decoder_loss_inf,
                           "decoder_r2_inf": model.decoder_r_squared_inf,
                           "encoder_r2": model.encoder_r_squared,
                           "s_decoder_val": s_decoder_val}

            # decoder training
            for epoch in range(self.n_epochs):
                # reset epoch in data set for each new epoch
                self.train_data.reset_epoch()

                for iteration in range(self.n_iterations):
                    enc_inp_batch, \
                    enc_seqlen, \
                    dec_tr_inp_batch, \
                    dec_inf_inp_batch, \
                    dec_seqlen, \
                    enc_tar_batch, \
                    dec_tar_batch = self.train_data.next_batch(enc_len=self.encoder_length,
                                                               dec_len=self.decoder_length,
                                                               batch_size=self.batch_size)

                    feed_dict = {model.encoder_inputs: enc_inp_batch,
                                 model.encoder_sequence_length: enc_seqlen,
                                 model.decoder_training_inputs: dec_tr_inp_batch,
                                 model.decoder_inference_inputs: dec_inf_inp_batch,
                                 model.decoder_sequence_length: dec_seqlen,
                                 model.encoder_targets: enc_tar_batch,
                                 model.decoder_targets: dec_tar_batch,
                                 model.input_keep_prob: self.input_keep_prob,
                                 model.output_keep_prob: self.output_keep_prob,
                                 model.state_keep_prob: self.state_keep_prob}

                    # execute training op and fetch evaluation values
                    vals = sess.run(train_fetches, feed_dict=feed_dict)

                    # log and evaluate after every epoch
                decoder_train_summary = vals["s_decoder_train"]
                global_step = vals["global_step_decoder_train"]
                decoder_r2_tr = vals["decoder_r2_inf"]  # inference r2 on training data set

                enc_inp_batch_val, \
                enc_seqlen_val, \
                dec_tr_inp_batch_val, \
                dec_inf_inp_batch_val, \
                dec_seqlen_val, \
                enc_tar_batch_val, \
                dec_tar_batch_val = self.val_data.next_batch(enc_len=self.encoder_length,
                                                             dec_len=self.decoder_length,
                                                             batch_size=self.val_data.n_sequences)

                feed_dict_val = {model.encoder_inputs: enc_inp_batch_val,
                                 model.encoder_sequence_length: enc_seqlen_val,
                                 model.decoder_training_inputs: dec_tr_inp_batch_val,
                                 model.decoder_inference_inputs: dec_inf_inp_batch_val,
                                 model.decoder_sequence_length: dec_seqlen_val,
                                 model.encoder_targets: enc_tar_batch_val,
                                 model.decoder_targets: dec_tar_batch_val}

                vals = sess.run(val_fetches, feed_dict=feed_dict_val)
                val_loss = vals["decoder_val_loss"]
                decoder_val_summary = vals["s_decoder_val"]

                if epoch % 10 == 0:
                    print("epoch: {} \t"
                          "step: {} \n"
                          "\ton validation set: "
                          "encoder_r2: {:.4f} \t"
                          "training_decoder_r2: {:.4f} \t"
                          "inference_decoder_r2: {:.4}".format(epoch, global_step,
                                                         vals["encoder_r2"], decoder_r2_tr, vals["decoder_r2_inf"]))

                if self.logging:
                    file_writer.add_summary(decoder_train_summary, global_step)
                    file_writer.add_summary(decoder_val_summary, global_step)

                if val_loss < best_loss:
                    best_params = self._get_model_params()
                    best_loss = val_loss
                    checks_wo_progress = 0
                else:
                    checks_wo_progress += 1

                if checks_wo_progress > self.max_checks_wo_progress:
                    print("Early stopping!")
                    break

            if best_params:
                self._restore_model_params(best_params)

            self.model = model

            return self

    def predict(self, X):
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)

        with self._session.as_default() as sess:

            enc_inp_batch_val, \
            enc_seqlen_val, \
            dec_tr_inp_batch_val, \
            dec_inf_inp_batch_val, \
            dec_seqlen_val, \
            enc_tar_batch_val, \
            dec_tar_batch_val = self.val_data.next_batch(enc_len=self.encoder_length,
                                                         dec_len=self.decoder_length,
                                                         batch_size=self.val_data.n_sequences)
                                                         # n_sequences for entire batch

            feed_dict = {self.model.encoder_inputs: enc_inp_batch_val,
                         self.model.encoder_sequence_length: enc_seqlen_val,
                         self.model.decoder_training_inputs: dec_tr_inp_batch_val,
                         self.model.decoder_inference_inputs: dec_inf_inp_batch_val,
                         self.model.decoder_sequence_length: dec_seqlen_val,
                         self.model.encoder_targets: enc_tar_batch_val,
                         self.model.decoder_targets: dec_tar_batch_val}

            fetches = {
                "y_pred_dec_tr": self.model.training_decoder_outputs,
                "y_pred_dec_inf": self.model.inference_decoder_outputs,
                "y_pred_enc": self.model.encoder_outputs,
                "dec_tar": self.model.decoder_targets,
                "enc_tar": self.model.encoder_targets,
                "r_squared_dec_inf": self.model.decoder_r_squared_inf,
                "r_squared_dec_tr": self.model.decoder_r_squared_tr,
                "r_squared_enc": self.model.encoder_r_squared
                # "encoder_rmse": self.model.encoder_rmse,
                # "decoder_rmse_inf": self.model.decoder_rmse_tr,
                # "decoder_rmse_tr": self.model.decoder_rmse_inf
                }
            # rmse, mae

            vals = sess.run(fetches, feed_dict)
            return vals

    def score(self, X, y=None):
        """ calculates R2 for scoring within sklearn grid/random search cv """
        vals = self.predict(X)
        return vals["r_squared_dec_inf"]

    def score_eval(self, X, y=None):
        vals = self.predict(X)

        dec_rmse_inf = self.rmse(vals["y_pred_dec_inf"], vals["dec_tar"])
        dec_rmse_tr = self.rmse(vals["y_pred_dec_tr"], vals["dec_tar"])
        enc_rmse = self.rmse(vals["y_pred_enc"], vals["enc_tar"])

        eval_dict = {"r2_dec_inf": vals["r_squared_dec_inf"],
                     "rmse_dec_inf": dec_rmse_inf,
                     "r2_dec_tr": vals["r_squared_dec_tr"],
                     "rmse_dec_tr": dec_rmse_tr,
                     "r2_enc": vals["r_squared_enc"],
                     "rmse_enc": enc_rmse}

        print("r2_dec_inf: {:.4f}, \t rmse_dec_inf: {:.4f}\n"
              "r2_dec_tr:  {:.4f}, \t rmse_dec_tr:  {:.4f}\n"
              "r2_enc:     {:.4f}, \t rmse_enc:     {:.4f}".format(
            vals["r_squared_dec_inf"], dec_rmse_inf, vals["r_squared_dec_tr"],
            dec_rmse_tr, vals["r_squared_enc"], enc_rmse))

        return eval_dict

    def inv_transform(self, X):
        """ inverse transforms an ndarray of shape [sequences, steps, depth]
        for backscaling of targets and outputs """

        sequences, steps, depth = X.shape
        X = X.reshape([-1, depth])

        X_backscaled = self.lab_pipeline.inverse_transform(X)
        X_backscaled = X_backscaled.reshape([sequences, steps, depth])

        return X_backscaled

    def rmse(self, outputs, targets):
        """ calculates rmse value """
        bs_targets = self.inv_transform(targets) # bs =^ backscaled
        bs_outputs = self.inv_transform(outputs)

        bs_loss = np.mean(np.square(bs_targets-bs_outputs))

        rmse = np.sqrt(bs_loss)
        return rmse




    # def infere(self):
    #     """ to be implemented, back-sclaing of labels using ds._lab_scaler""
    #     return None

    def save(self, path):
        self._saver.save(self._session, path)


class seq2seqEncoderConfig(BaseEstimator, ClassifierMixin):
    """ 
    
    """

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 train_data,
                 val_data,
                 n_layers=2,
                 n_neurons=20,
                 batch_size=30,
                 cell_type="lstm-layer-norm",
                 activation_fn=tf.tanh,
                 decoder_length=24,
                 encoder_length=48,
                 learning_rate=0.01,
                 gradient_clip=1.0,
                 input_keep_prob=0.8,
                 output_keep_prob=0.8,
                 state_keep_prob=1,
                 label_keep_prob=0.8,  # to be implemented
                 eos_token=-1,
                 initializer=tf.contrib.layers.variance_scaling_initializer(),
                 n_epochs=250,
                 max_checks_wo_progress=100,
                 logging=True,
                 base_path=None):
        """ Initializer:

        
        """

        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.cell_type = cell_type
        self.activation_fn = activation_fn
        self.decoder_length = decoder_length
        self.encoder_length = encoder_length
        self.learning_rate = learning_rate
        self.gradient_clip = gradient_clip
        self.input_keep_prob = input_keep_prob
        self.output_keep_prob = output_keep_prob
        self.state_keep_prob = state_keep_prob
        self.label_keep_prob = label_keep_prob
        self.eos_token = eos_token
        self.initializer = initializer
        self.n_epochs = n_epochs
        self.max_checks_wo_progress = max_checks_wo_progress
        self.logging = logging
        self.base_path = base_path

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.train_data = train_data
        self.val_data = val_data
        self.n_iterations = self.train_data.get_iterations(batch_size=self.batch_size)

        if train_data.prep_pipeline_lab:
            self.lab_pipeline = train_data.lab_scaler

        self._session = None
        self._graph = None
        self.encoder_train_model = None

    def _close_session(self):
        """ closes current session when called """
        if self._session is not None:
            self._session.close()

    def _get_model_params(self):
        """ Get all variable values """
        with self._graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value for gvar, value in zip(gvars, self._session.run(gvars))}

    def _restore_model_params(self, model_params):
        """ Set all variables to the given values (for early stopping, faster than loading from disk)"""
        gvar_names = list(model_params.keys())
        assign_ops = {gvar_name: self._graph.get_operation_by_name(gvar_name + "/Assign")
                      for gvar_name in gvar_names}
        init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
        feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
        self._session.run(assign_ops, feed_dict=feed_dict)

    def fit(self, X, y=None):
        """
        fit method as described in sklearn baseestimator. 
        """

        self._close_session()

        now = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        self.log_dir = "{}/encoder/run-{}".format(self.base_path, now)
        self.save_path = "{}/encoder/run-{}/encoder/encoder_ckpt.ckpt".format(self.base_path, now)

        print("encoder run-{}".format(now))

        self._graph = tf.Graph()
        with self._graph.as_default() as graph:

            # build model and summaries

            encoder_train_model = seq2seqModel(config=self)

            tf.summary.scalar("encoder_loss",
                              encoder_train_model.encoder_loss,
                              collections=["encoder_training", "encoder_validation"])
            tf.summary.scalar("encoder_r2",
                              encoder_train_model.encoder_r_squared,
                              collections=["encoder_training", "encoder_validation"])
            tf.summary.histogram("encoder_weights",
                                 encoder_train_model.encoder_weights,
                                 collections=["encoder_validation"])
            tf.summary.histogram("encoder_bias",
                                 encoder_train_model.encoder_bias,
                                 collections=["encoder_validation"])
            s_encoder_train = tf.summary.merge_all("encoder_training")
            s_encoder_val = tf.summary.merge_all("encoder_validation")

            # initialize graph
            init = tf.global_variables_initializer()
            self._saver = tf.train.Saver()

        # early stopping
        checks_wo_progress = 0
        best_loss = np.infty
        best_params = None

        self._session = tf.Session(graph=graph)
        with self._session.as_default() as sess:
            init.run()

            if self.logging:
                file_writer = tf.summary.FileWriter(self.log_dir, graph=graph)

            train_fetches = {"train_op": encoder_train_model.encoder_train_op,
                             "global_step_encoder_train": encoder_train_model.global_step_encoder,
                             "encoder_r2": encoder_train_model.encoder_r_squared,
                             "s_encoder_train": s_encoder_train}
            val_fetches = {"global_step_encoder_train": encoder_train_model.global_step_encoder,
                           "encoder_val_loss": encoder_train_model.encoder_loss,
                           "encoder_r2": encoder_train_model.encoder_r_squared,
                           "s_encoder_val": s_encoder_val}

            # encoder training
            for epoch in range(self.n_epochs):
                # reset epoch in data set for each new epoch
                self.train_data.reset_epoch()

                for iteration in range(self.n_iterations):
                    enc_inp_batch, \
                    enc_seqlen, \
                    dec_tr_inp_batch, \
                    dec_inf_inp_batch, \
                    dec_seqlen, \
                    enc_tar_batch, \
                    dec_tar_batch = self.train_data.next_batch(enc_len=self.encoder_length,
                                                               dec_len=self.decoder_length,
                                                               batch_size=self.batch_size)

                    feed_dict = {encoder_train_model.encoder_inputs: enc_inp_batch,
                                 encoder_train_model.encoder_sequence_length: enc_seqlen,
                                 encoder_train_model.decoder_training_inputs: dec_tr_inp_batch,
                                 encoder_train_model.decoder_inference_inputs: dec_inf_inp_batch,
                                 encoder_train_model.decoder_sequence_length: dec_seqlen,
                                 encoder_train_model.encoder_targets: enc_tar_batch,
                                 encoder_train_model.decoder_targets:dec_tar_batch,
                                 encoder_train_model.input_keep_prob: self.input_keep_prob,
                                 encoder_train_model.output_keep_prob: self.output_keep_prob,
                                 encoder_train_model.state_keep_prob: self.state_keep_prob}

                    # execute training op and fetch evaluation values
                    vals = sess.run(train_fetches, feed_dict=feed_dict)

                # log and evaluate after every epoch
                encoder_train_summary = vals["s_encoder_train"]
                global_step = vals["global_step_encoder_train"]
                encoder_r2_tr = vals["encoder_r2"]  # encoder r2 on train data set

                enc_inp_batch_val, \
                enc_seqlen_val, \
                dec_tr_inp_batch_val, \
                dec_inf_inp_batch_val, \
                dec_seqlen_val, \
                enc_tar_batch_val, \
                dec_tar_batch_val = self.val_data.next_batch(enc_len=self.encoder_length,
                                                             dec_len=self.decoder_length,
                                                             batch_size=self.val_data.n_sequences)

                feed_dict_val = {encoder_train_model.encoder_inputs: enc_inp_batch_val,
                                 encoder_train_model.encoder_sequence_length: enc_seqlen_val,
                                 encoder_train_model.decoder_training_inputs: dec_tr_inp_batch_val,
                                 encoder_train_model.decoder_inference_inputs: dec_inf_inp_batch_val,
                                 encoder_train_model.decoder_sequence_length: dec_seqlen_val,
                                 encoder_train_model.encoder_targets: enc_tar_batch_val,
                                 encoder_train_model.decoder_targets:dec_tar_batch_val}

                vals = sess.run(val_fetches, feed_dict=feed_dict_val)
                val_loss = vals["encoder_val_loss"]
                encoder_val_summary = vals["s_encoder_val"]

                if epoch % 10 == 0:
                    print("epoch: {} \t"
                          "step: {} \n"
                          "\ton training set batch: "
                          "encoder_r2_tr: {:.4f} \t"
                          "on entire validation set: "
                          "encoder_r2_val: {:.4}".format(epoch, global_step, encoder_r2_tr, vals["encoder_r2"]))

                if self.logging:
                    file_writer.add_summary(encoder_train_summary, global_step)
                    file_writer.add_summary(encoder_val_summary, global_step)

                if val_loss < best_loss:
                    best_params = self._get_model_params()
                    best_loss = val_loss
                    checks_wo_progress = 0
                else:
                    checks_wo_progress += 1

                if checks_wo_progress > self.max_checks_wo_progress:
                    print("Early stopping!")
                    break

            if best_params:
                self._restore_model_params(best_params)

            self.encoder_train_model = encoder_train_model

            return self

    def predict(self, X):
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)

        with self._session.as_default() as sess:

            enc_inp_batch_val, \
            enc_seqlen_val, \
            dec_tr_inp_batch_val, \
            dec_inf_inp_batch_val, \
            dec_seqlen_val, \
            enc_tar_batch_val, \
            dec_tar_batch_val = self.val_data.next_batch(enc_len=self.encoder_length,
                                                         dec_len=self.decoder_length,
                                                         batch_size=self.val_data.n_sequences)
                                                         # n_sequences for entire batch

            feed_dict = {self.encoder_train_model.encoder_inputs: enc_inp_batch_val,
                         self.encoder_train_model.encoder_sequence_length: enc_seqlen_val,
                         self.encoder_train_model.decoder_training_inputs: dec_tr_inp_batch_val,
                         self.encoder_train_model.decoder_inference_inputs: dec_inf_inp_batch_val,
                         self.encoder_train_model.decoder_sequence_length: dec_seqlen_val,
                         self.encoder_train_model.encoder_targets: enc_tar_batch_val,
                         self.encoder_train_model.decoder_targets: dec_tar_batch_val}

            fetches = {
                "y_pred": self.encoder_train_model.encoder_outputs,
                "r_squared": self.encoder_train_model.encoder_r_squared
            }

            vals = sess.run(fetches, feed_dict)
            return vals

    def score(self, X, y=None):
        """ calculates R2 for scoring within sklearn grid/random search cv """
        vals = self.predict(X)
        return vals["r_squared"]


class seq2seqSearch(object):
    """
    
    """

    def __init__(self, arch_search_params, train_search_params, data, encoder_length, decoder_length):
        """
        
            args:
        arch_search_params: search params that define the architecture, i.e. layers, neurons
        train_search_params: search params that define the training, i.e. learning rate, grad clip, batch_size
        data: a data_set.DataPreparation object
        """

        self.arch_search_params = arch_search_params
        self.train_search_params = train_search_params

        self.merged_params = {**self.arch_search_params, **self.train_search_params}  # merged params: encoder training

        seqlen = encoder_length+decoder_length
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.train_data, self.val_data, self.test_data = data.split_data(model="seq2seq",
                                                                         sequence_length=seqlen,
                                                                         train_ratio=0.80,
                                                                         val_ratio=0.10,
                                                                         test_ratio=0.10)
        self.n_inputs = self.train_data.n_features
        self.n_outputs = self.train_data.n_labels

        self._encoder_model = None
        self._decoder_model = None

    def search(self, logging, base_path, n_iter, dec_n_iter, cv, epochs, max_checks_wo_progress, seed=42):
        """
        
        """

        X_val = np.arange(10)

        self._encoder_model = seq2seqEncoderConfig(n_inputs=self.n_inputs, n_outputs=self.n_outputs,
                                                   train_data=self.train_data, val_data=self.val_data,
                                                   n_epochs=epochs, max_checks_wo_progress=max_checks_wo_progress,
                                                   encoder_length=self.encoder_length, decoder_length=self.decoder_length,
                                                   logging=logging, base_path=base_path)

        encoder_rnd_search = RandomizedSearchCV(self._encoder_model, param_distributions=self.merged_params,
                                                n_iter=n_iter, cv=cv, verbose=2, random_state=None)
        encoder_rnd_search.fit(X_val)

        # print best r2 of encoder with which params, ask if decoder shall be trained with given best model
        # if yes save and load model in decoder training (decoder training mus have same architecture as encoder
        # train decoder, return best results, ask if shall save model

        best_encoder_fit = encoder_rnd_search.best_estimator_.score(X_val)
        rnd_search_params = encoder_rnd_search.best_params_
        best_encoder_params = encoder_rnd_search.best_estimator_._get_model_params()

        print("best encoder found has r2: {:.4f}".format(best_encoder_fit))
        print("best encoder uses these parameters: {}".format(rnd_search_params))

        retrain_w_hparams_q = input("retrain encoder with best hparams (yes/no)? ")
        while retrain_w_hparams_q not in ["yes", "no"]:
            retrain_w_hparams_q = input("retrain enoder with best hparams (yes/no)? ")

        while retrain_w_hparams_q=="yes":
            self._encoder_model = seq2seqEncoderConfig(n_inputs=self.n_inputs,
                                                        n_outputs=self.n_outputs,
                                                        train_data=self.train_data,
                                                        val_data=self.val_data,
                                                        n_epochs=epochs,
                                                        max_checks_wo_progress=max_checks_wo_progress,
                                                        encoder_length=self.encoder_length,
                                                        decoder_length=self.decoder_length,
                                                        logging=logging, base_path=base_path,
                                                        n_layers=rnd_search_params["n_layers"],
                                                        n_neurons=rnd_search_params["n_neurons"],
                                                        batch_size=rnd_search_params["batch_size"],
                                                        cell_type=rnd_search_params["cell_type"],
                                                        learning_rate=rnd_search_params["learning_rate"],
                                                        gradient_clip=rnd_search_params["gradient_clip"],
                                                        input_keep_prob=rnd_search_params["input_keep_prob"],
                                                        output_keep_prob=rnd_search_params["output_keep_prob"],
                                                        state_keep_prob=rnd_search_params["state_keep_prob"])

            self._encoder_model.fit(X_val)

            print("found encoder r2: {:.4f}".format(self._encoder_model.score(X_val)))

            overwrite_q = input("overwrite existing best encoder (yes/no?) ")
            while overwrite_q not in ["yes", "no"]:
                overwrite_q = input("overwrite existing best encoder (yes/no?) ")
            if overwrite_q=="yes":
                best_encoder_params = self._encoder_model._get_model_params()

            retrain_w_hparams_q = input("retrain enoder with best hparams (yes/no?) ")
            while retrain_w_hparams_q not in ["yes", "no"]:
                retrain_w_hparams_q = input("retrain enocder with best hparams (yes/no?) ")

        train_decoder_q = input("start decoder training (yes/no)? ")
        while train_decoder_q not in ["yes", "no"]:
            train_decoder_q = input("start decoder training (yes/no)? ")

        if train_decoder_q=="yes":

            self._decoder_model = seq2seqDecoderConfig(n_inputs=self.n_inputs, n_outputs=self.n_outputs,
                                                       train_data=self.train_data, val_data=self.val_data,
                                                       logging=logging, base_path=base_path,
                                                       config=encoder_rnd_search.best_estimator_,
                                                       model_restore_params=best_encoder_params)

            decoder_rnd_search = RandomizedSearchCV(self._decoder_model,
                                                    param_distributions=self.train_search_params,
                                                    n_iter=dec_n_iter, cv=cv, verbose=2, random_state=seed)
            decoder_rnd_search.fit(X_val, model_restore_params=best_encoder_params)

            best_decoder_fit = decoder_rnd_search.best_estimator_.score(X_val)

            self._decoder_model = decoder_rnd_search.best_estimator_

            print("best encoder-decoder model found has r2: {:.4f}".format(best_decoder_fit))

            save_model_q = input("save model (yes/no)? ")
            while save_model_q not in ["yes", "no"]:
                save_model_q = input("save model (yes/no)? ")

            if save_model_q=="yes":
                save_path_ext = input("extension of base_path to store model (w/o first /) ")
                save_path = str(base_path + '/' + save_path_ext + '/model.ckpt')
                decoder_rnd_search.best_estimator_.save(path=save_path)
                file = open(str(base_path + '/' + save_path_ext + '/config.txt'), "w")
                file.write(str(rnd_search_params))
                file = open(str(base_path + '/' + save_path_ext + '/final_eval.txt'), "w")
                file.write(str(self._decoder_model.score_eval(X_val)))

            retrain_enc_dec = input("retrain encoder-decoder model (yes/no)? ")
            while retrain_enc_dec not in ["yes", "no"]:
                retrain_enc_dec = input("retrain encoder-decoder model (yes/no)? ")
            while retrain_enc_dec=="yes":

                self._decoder_model = seq2seqDecoderConfig(n_inputs=self.n_inputs, n_outputs=self.n_outputs,
                                                           train_data=self.train_data, val_data=self.val_data,
                                                           logging=logging, base_path=base_path,
                                                           config=decoder_rnd_search.best_estimator_,
                                                           model_restore_params=best_encoder_params)

                self._decoder_model.fit(X_val, model_restore_params=best_encoder_params)

                best_decoder_fit = decoder_rnd_search.best_estimator_.score(X_val)

                print("best encoder-decoder model found has r2: {:.4f}".format(best_decoder_fit))

                save_model_q = input("save model (only when logging was enabled!!) (yes/no)? ")
                while save_model_q not in ["yes", "no"]:
                    save_model_q = input("save model (yes/no)? ")

                if save_model_q == "yes":
                    save_path_ext = input("extension of base_path to store model (w/o first /) ")
                    save_path = str(base_path + '/' + save_path_ext + '/model.ckpt')
                    decoder_rnd_search.best_estimator_.save(path=save_path)
                    file = open(str(base_path + '/' + save_path_ext + '/config.txt'), "w")
                    file.write(str(rnd_search_params))
                    file = open(str(base_path + '/' + save_path_ext + '/final_eval.txt'), "w")
                    file.write(self._decoder_model.score_eval(X_val))

                retrain_enc_dec = input("retrain encoder-decoder model (yes/no)? ")
                while retrain_enc_dec not in ["yes", "no"]:
                    retrain_enc_dec = input("retrain encoder-decoder model (yes/no)? ")

        self._decoder_model.score_eval(X_val)
        if logging:
            file = open(str(base_path + '/' + '/final_eval.txt'), "w")
            file.write(str(self._decoder_model.score_eval(X_val)));

        print("finished")

        return self._decoder_model
