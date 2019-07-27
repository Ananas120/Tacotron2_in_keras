import tensorflow as tf
import keras.backend as K

from keras.layers import *

from tacotron.models.ArchitectureWrappers import *
from tacotron.models.modules import *
from tacotron.models.new_layers import *


def build_tacotron(hp):
    is_training = True
    post_condition = True

    input_text = Input(shape=(None,))
    embedding = Embedding(input_dim=100, output_dim=128)(input_text)

    encoder_cell = TacotronEncoderCell(
        EncoderConvolutions(is_training, hp),
        EncoderRNN(is_training, size=hp.encoder_lstm_units,
                  zoneout=hp.tacotron_zoneout_rate)
    )

    encoder_outputs = encoder_cell(embedding)

    input_decoder = Input(shape=(None, hp.num_mels))

    prenet = PreNet(is_training, layers_sizes=hp.prenet_layers, drop_rate=hp.tacotron_dropout_rate)

    decoder_lstm = DecoderRNN(is_training, layers=hp.decoder_layers, size=hp.decoder_lstm_units)

    attention_layer = LocationSensitiveAttentionLayer(encoder_outputs, hp.attention_dim, filters=hp.attention_filters, kernel=hp.attention_kernel, rnn_cell=decoder_lstm, cumulate_weights=hp.cumulative_weights)

    frame_projection = FrameProjection(hp.num_mels, name='decoder_output')

    stop_projection = StopProjection(is_training)



    decoder_cell = TacotronDecoderCell(
        prenet,
        attention_layer,
        frame_projection,
        stop_projection
    )
    (frame_prediction, stop_prediction) = decoder_cell([encoder_outputs, input_decoder], debug=True)

    decoder_output = frame_prediction

    postnet = PostNet(is_training, hparams=hp, name='postnet_convolutions')

    residual = postnet(decoder_output)

    residual_projection = FrameProjection(hp.num_mels, name='postnet_projection')
    projected_residual = residual_projection(residual)

    mel_outputs = Add(name='mel_predictions')([decoder_output, projected_residual])

    if post_condition: 
        post_cbhg = CBHG(hp.cbhg_kernels, hp.cbhg_conv_channels, hp.cbhg_pool_size, [hp.cbhg_projection, hp.num_mels],
                        hp.cbhg_projection_kernel_size, hp.cbhg_highwaynet_layers,
                        hp.cbhg_highway_units, hp.cbhg_rnn_units, hp.batch_norm_position, is_training, name='CBHG_postnet')

        post_outputs = post_cbhg(mel_outputs, debug=True)

        linear_specs_projection = FrameProjection(hp.num_freq, name='linear_spectrogram_projection')

        linear_outputs = linear_specs_projection(post_outputs)

    inputs = (input_text, input_decoder)
    outputs = (mel_outputs, linear_outputs, stop_prediction)
    model = Model(inputs, outputs)
    model.summary(150)
    return model