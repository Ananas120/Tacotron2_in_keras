import keras
import tensorflow as tf
import keras.backend as K

from keras.layers import *
from tensorflow.python.framework.tensor_shape import *
from tensorflow.python.ops import rnn_cell_impl, array_ops, math_ops

class ZoneoutLSTMCell(LSTMCell):
    def __init__(self, units, is_training=False, 
                 zoneout_factor_cell=0., 
                 zoneout_factor_output=0., 
                 **kwargs):
        super(ZoneoutLSTMCell, self).__init__(units, **kwargs)
        zm = min(zoneout_factor_output, zoneout_factor_cell)
        zs = max(zoneout_factor_output, zoneout_factor_cell)
        
        if zm < 0. or zs > 1.:
            raise ValueError('One / both provided zoneoutfacotrs are not in [0, 1]')
           
        self._zoneout_cell = zoneout_factor_cell
        self._zoneout_outputs = zoneout_factor_output
                
    def get_config(self):
        config = super(ZoneoutLSTMCell, self).get_config()
        config['zoneout_factor_cell'] = self._zoneout_cell
        config['zoneout_factor_output'] = self._zoneout_outputs
        return config
                
    def call(self, inputs, state, **kwargs):
        output, new_state = super(ZoneoutLSTMCell, self).call(inputs, state)
        (prev_c, prev_h) = state
        (new_c, new_h) = new_state

        drop_c = K.dropout(new_c - prev_c, self._zoneout_cell)
        drop_h = K.dropout(new_h - prev_h, self._zoneout_outputs)
        c = (1. - self._zoneout_cell) * drop_c + prev_c
        h = (1. - self._zoneout_outputs) * drop_h + prev_h
            
        return output, [c, h]


        
class LocationSensitiveAttentionLayer(Layer):
    def __init__(self, units, filters, rnn_cell=None, kernel=3, smoothing=False, cumulate_weights=True, **kwargs):
        super(LocationSensitiveAttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.filters = filters
        self._cumulate = cumulate_weights
        
        self.location_convolution = Conv1D(filters=filters, kernel_size=kernel, padding='same', bias_initializer='zeros', name='location_features_convolution')
        self.location_layer = Dense(units, use_bias=False, name='location_features_layer')
        self.query_layer = Dense(units, use_bias=False)
        self.memory_layer = Dense(units, use_bias=False)
        self.rnn_cell = rnn_cell
        
        self.values = None #memory
        self.keys = None #self.memory_layer(self.values) if self.memory_layer else self.values
    
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        enc_out_seq, dec_out_seq = input_shape
        self.v_a = self.add_weight(name='V_a',
                                   shape=(self.units,),
                                   initializer='uniform',
                                   trainable=True)
        self.b_a = self.add_weight(name='b_a',
                                   shape=(self.units,),
                                   initializer='uniform',
                                   trainable=True)
        if self.memory_layer:
            self.memory_layer.build(enc_out_seq)
            self._trainable_weights += self.memory_layer._trainable_weights
        if self.query_layer:
            if not self.query_layer.built: 
                if self.rnn_cell:
                    self.query_layer.build(self.rnn_cell.compute_output_shape(dec_out_seq)[0])
                else:
                    self.query_layer.build(dec_out_seq)
            self._trainable_weights += self.query_layer._trainable_weights
        if self.rnn_cell:
            rnn_input_shape = (enc_out_seq[0], 1, dec_out_seq[-1] + enc_out_seq[-1])
            self.rnn_cell.build(rnn_input_shape)
            self._trainable_weights += self.rnn_cell.weights
            
        
        conv_input_shape = (enc_out_seq[0], enc_out_seq[1], 1)
        location_input_shape = (enc_out_seq[0], enc_out_seq[1], self.filters)
        self.location_convolution.build(conv_input_shape)
        self.location_layer.build(location_input_shape)
        
        self._trainable_weights += self.location_convolution._trainable_weights
        self._trainable_weights += self.location_layer._trainable_weights
        
        super(LocationSensitiveAttentionLayer, self).build(input_shape)
    
    def call(self, inputs, verbose=False):
        """
        Inputs : [encoder_output_sequence, decoder_input_sequence]
        decoder_input_sequence = last_decoder_output_sequence
        """
        assert isinstance(inputs, list)
        encoder_out_seq, decoder_out_seq = inputs
        #if self.values is None or self.values is not encoder_out_seq:
        #    print("Remplacing self.values : {} by : {}".format(self.values, encoder_out_seq))
        #    self.values = encoder_out_seq
        #    self.keys = self.memory_layer(self.values) if self.memory_layer else self.values
        values = encoder_out_seq
        keys = self.memory_layer(values) if self.memory_layer else values
        if verbose:
            print("encoder_out_seq shape (batch_size, input_timesteps, encoder_size): {}".format(encoder_out_seq.shape))
            print("decoder_out_seq shape (batch_size, last_outputs_timesteps, decoder_size): {}".format(decoder_out_seq.shape))
      
    
        def energy_step(query, states):
            """ Step function for computing energy for a single decoder state """""
            assert_msg = "States must be a list. However states {} is of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            """ Some parameters required for shaping Tensors """""

            previous_alignments = states[0]
            if self.rnn_cell:
                c_i = states[1]
                cell_state = states[2:]
                if verbose:
                    print("c_i : {}".format(c_i.shape))
                    print("cell_state : {}".format(cell_state))
            
                lstm_input = K.concatenate([query, c_i])
                lstm_input = K.expand_dims(lstm_input, 1)
                if verbose:
                    print("cell_state : {}".format(cell_state))
                    print("lstm_input : {}".format(lstm_input))
                lstm_out = self.rnn_cell(lstm_input, initial_state=cell_state)
                lstm_output, new_cell_state = lstm_out[0], lstm_out[1:]
                query = lstm_output
            
            processed_query = self.query_layer(query) if self.query_layer else query
            
            expanded_alignments = K.expand_dims(previous_alignments, axis=2)
            
            f = self.location_convolution(expanded_alignments)
            
            processed_location_features = self.location_layer(f)
            
            if verbose:
                print("query : {}".format(query.shape))
                print("previous_alignments : {}".format(previous_alignments.shape))
                print("processed_query : {}".format(processed_query.shape))
                print("f : {}".format(f.shape))
                print("processed_location_features : {}".format(processed_location_features.shape))
                
            
            e_i = K.sum(self.v_a * K.tanh(keys + processed_query + processed_location_features + self.b_a), [2])
            e_i = K.softmax(e_i) #K.exp(e_i) / K.sum(K.exp(e_i)) + K.epsilon()
            
            if self._cumulate:
                next_state = e_i + previous_alignments
            else:
                next_state = e_i

            if verbose:
                print("E_i : {}".format(e_i.shape))
            
            if self.rnn_cell:
                new_c_i, _ = context_step(e_i, [c_i])
            
                return e_i, [next_state, new_c_i, *new_cell_state]
            return e_i, [next_state]

        def context_step(inputs, states):
            """ Step function for computing c_i using e_i """""
            
            alignments = inputs
            expanded_alignments = K.expand_dims(alignments, 1)
            
            if verbose:
                print("expanded_alignments : {}".format(expanded_alignments.shape))
            
            c_i = math_ops.matmul(expanded_alignments, values)
            c_i = K.squeeze(c_i, 1)
            
            if verbose:
                print("C_i : {}".format(c_i.shape))
            return c_i, [c_i]

        def create_initial_state(inputs, hidden_size):
            fake_state = K.zeros_like(inputs)
            fake_state = K.sum(fake_state, axis=[1, 2])
            fake_state = K.expand_dims(fake_state)
            fake_state = K.tile(fake_state, [1, hidden_size])
            return fake_state
          
        def get_fake_cell_input(fake_state_c):
            fake_input = K.zeros_like(decoder_out_seq)[:,0,:]
            fake_input = K.concatenate([fake_state_c, fake_input])
            fake_input = K.expand_dims(fake_input, 1)
            return fake_input

        fake_state_c = create_initial_state(values, values.shape[-1])
        fake_state_e = create_initial_state(values, K.shape(values)[1])
        if self.rnn_cell:
            cell_initial_state = self.rnn_cell.get_initial_state(get_fake_cell_input(fake_state_c))
            initial_states_e = [fake_state_e, fake_state_c, *cell_initial_state]
        else:
            initial_states_e = [fake_state_e]
        if verbose:
            print("fake_state_c : {}".format(fake_state_c.shape))
            print("fake_state_e : {}".format(fake_state_e.shape))

        """ Computing energy outputs """""
        last_out, e_outputs, _ = K.rnn(energy_step,
                                       decoder_out_seq, 
                                       initial_states_e)
        """ Computing context vectors """""
        #last_out, c_outputs, _ = K.rnn(context_step,
        #                               e_outputs,
        #                               [fake_state_c])
        c_outputs = math_ops.matmul(e_outputs, values)


        if verbose:
            print("energy outputs : {}".format(e_outputs.shape))
            print("context vectors : {}".format(c_outputs.shape))

        return [c_outputs, e_outputs]
  

    def comute_output_shape(self, input_shape):
        """ Outputs produced by the layer """""
        return [
            (input_shape[1][0], input_shape[1][1], input_shape[1][2]),
            (input_shape[1][0], input_shape[1][1], input_shape[0][1])
        ]
