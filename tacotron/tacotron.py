import tensorflow as tf
import keras.backend as K

from keras.layers import *

from tacotron.models.ArchitectureWrappers import *
from tacotron.models.modules import *
from tacotron.models.new_layers import *


def build_tacotron(hp):
    is_training = True

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

    attention_layer = LocationSensitiveAttentionLayer(hp.attention_dim, filters=hp.attention_filters, kernel=hp.attention_kernel, rnn_cell=decoder_lstm, cumulate_weights=hp.cumulative_weights)

    frame_projection = FrameProjection(hp.num_mels)

    stop_projection = StopProjection(is_training)

    decoder_cell = TacotronDecoderCell(
        prenet,
        attention_layer,
        frame_projection,
        stop_projection
    )
    (frame_prediction, stop_prediction) = decoder_cell([encoder_outputs, input_decoder])

    inputs = (input_text, input_decoder)
    outputs = (frame_prediction, stop_prediction)
    model = Model(inputs, outputs)
    model.summary()
    return model