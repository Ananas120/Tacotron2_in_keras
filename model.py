import tensorflow as tf
import keras.backend as K

from keras.layers import *
from keras.models import *
from tensorflow.python.ops import array_ops, math_ops, check_ops, rnn_cell_impl, tensor_array_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.util import nest

from modules import *
from new_layers import *


class TacotronEncoderCell:
    def __init__(self, convolutional_layers, lstm_layer):
        self._convolutions = convolutional_layers
        self._cell = lstm_layer
        
    def __call__(self, inputs, input_lengths=None):
        conv_output = self._convolutions(inputs)

        hidden_representation = self._cell(conv_output)
        
        self.conv_output_shape = conv_output.shape
        
        return hidden_representation
                                                                                      
    
class TacotronDecoderCell:
    def __init__(self, prenet, attention_mechanism, frame_projection, stop_projection):                
        self._prenet = prenet
        self._attention = attention_mechanism
        self._frame_projection = frame_projection
        self._stop_projection = stop_projection
                
    
    
    def __call__(self, inputs, debug=False):
        encoder_out_seq, decoder_out_seq = inputs
        prenet_output = self._prenet(decoder_out_seq)
                        
        
        context_vector, alignments = self._attention([encoder_out_seq, prenet_output], verbose=debug)
        
        projections_input = Concatenate(axis=-1)([prenet_output, context_vector])
        
        cell_outputs = self._frame_projection(projections_input)
        stop_tokens = self._stop_projection(projections_input)
                        
        return cell_outputs, stop_tokens, alignments
    
def build_tacotron_model(hp, vocab_size, is_training=True, debug=False):
    output_range = (-hp.max_abs_value, hp.max_abs_value) if hp.symmetric_mels else (0, hp.max_abs_value)
    
    input_text = Input(shape=(None,), name="Encoder_input")
    embedding = Embedding(input_dim=vocab_size, output_dim=hp.embedding_dim)(input_text)

    encoder_cell = TacotronEncoderCell(
        EncoderConvolutions(is_training, hp),
        EncoderRNN(is_training, size=hp.encoder_lstm_units,
                  zoneout=hp.tacotron_zoneout_rate)
    )

    encoder_outputs = encoder_cell(embedding)

    input_decoder_from_encoder = Input(shape=(None, 2 * hp.encoder_lstm_units), name="Decoder_input_from_encoder")
    input_decoder = Input(shape=(None, hp.num_mels * hp.outputs_per_step), name="Decoder_last_outputs")

    prenet = PreNet(is_training, layers_sizes=hp.prenet_layers, drop_rate=hp.tacotron_dropout_rate)

    decoder_lstm = DecoderRNN(is_training, layers=hp.decoder_layers, size=hp.decoder_lstm_units)

    attention_layer = LocationSensitiveAttentionLayer(hp.attention_dim, filters=hp.attention_filters, kernel=hp.attention_kernel, rnn_cell=decoder_lstm, cumulate_weights=hp.cumulative_weights)

    frame_projection = FrameProjection(hp.num_mels * hp.outputs_per_step, name='decoder_output')

    stop_projection = StopProjection(is_training, shape=hp.outputs_per_step)


    decoder_cell = TacotronDecoderCell(
        prenet,
        attention_layer,
        frame_projection,
        stop_projection
    )
    (frame_prediction, stop_prediction, weights) = decoder_cell([input_decoder_from_encoder, input_decoder], debug=debug)

    decoder_output = Reshape((-1, hp.num_mels))(frame_prediction)
    stop_prediction = Reshape((-1, 1))(stop_prediction)

    postnet = PostNet(is_training, hparams=hp, name='postnet_convolutions')

    residual = postnet(decoder_output)

    residual_projection = FrameProjection(hp.num_mels, name='postnet_projection')
    projected_residual = residual_projection(residual)

    mel_outputs = Add(name='mel_predictions')([decoder_output, projected_residual])

    if hp.clip_outputs:
        mel_outputs = Lambda(lambda x: K.clip(x, min_value=output_range[0], max_value=output_range[1]))(mel_outputs)
    
    if hp.predict_linear: 
        post_cbhg = CBHG(hp.cbhg_kernels, hp.cbhg_conv_channels, hp.cbhg_pool_size, [hp.cbhg_projection, hp.num_mels],
                        hp.cbhg_projection_kernel_size, hp.cbhg_highwaynet_layers,
                        hp.cbhg_highway_units, hp.cbhg_rnn_units, hp.batch_norm_position, is_training, name='CBHG_postnet')

        post_outputs = post_cbhg(mel_outputs, debug=debug)

        linear_specs_projection = FrameProjection(hp.num_freq, name='linear_spectrogram_projection')

        linear_outputs = linear_specs_projection(post_outputs)

        if hp.clip_outputs:
            linear_outputs = Lambda(lambda x: K.clip(x, min_value=output_range[0], max_value=output_range[1]))(linear_outputs)

    encoder_model = Model(inputs=input_text, outputs=encoder_outputs, name="Encoder")

    decoder_model = Model(inputs=[input_decoder_from_encoder, input_decoder], 
                          outputs=[decoder_output, mel_outputs, linear_outputs, stop_prediction, weights], 
                          name="Decoder")

    return encoder_model, decoder_model