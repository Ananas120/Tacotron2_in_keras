import tensorflow as tf
import keras.backend as K

from keras.layers import *

from tacotron.models.ArchitectureWrappers import *
from tacotron.models.modules import *
from tacotron.models.new_layers import *


def build_tacotron(hp):
    is_training = True
    post_condition = True

    input_text = Input(shape=(None,), name="Encoder_input")
    embedding = Embedding(input_dim=100, output_dim=128)(input_text)

    encoder_cell = TacotronEncoderCell(
        EncoderConvolutions(is_training, hp),
        EncoderRNN(is_training, size=hp.encoder_lstm_units,
                  zoneout=hp.tacotron_zoneout_rate)
    )

    encoder_outputs = encoder_cell(embedding)

    input_decoder_from_encoder = Input(shape=(None, 2 * hp.encoder_lstm_units), name="Decoder_input_from_encoder")
    input_decoder = Input(shape=(None, hp.num_mels), name="Decoder_last_outputs")

    prenet = PreNet(is_training, layers_sizes=hp.prenet_layers, drop_rate=hp.tacotron_dropout_rate)

    decoder_lstm = DecoderRNN(is_training, layers=hp.decoder_layers, size=hp.decoder_lstm_units)

    attention_layer = LocationSensitiveAttentionLayer(input_decoder_from_encoder, hp.attention_dim, filters=hp.attention_filters, kernel=hp.attention_kernel, rnn_cell=decoder_lstm, cumulate_weights=hp.cumulative_weights)

    frame_projection = FrameProjection(hp.num_mels, name='decoder_output')

    stop_projection = StopProjection(is_training)


    decoder_cell = TacotronDecoderCell(
        prenet,
        attention_layer,
        frame_projection,
        stop_projection
    )
    (frame_prediction, stop_prediction, weights) = decoder_cell([input_decoder_from_encoder, input_decoder], debug=False)

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

        post_outputs = post_cbhg(mel_outputs, debug=False)

        linear_specs_projection = FrameProjection(hp.num_freq, name='linear_spectrogram_projection')

        linear_outputs = linear_specs_projection(post_outputs)


    encoder_model = Model(inputs=input_text, outputs=encoder_outputs, name="Encoder")

    decoder_model = Model(inputs=[input_decoder_from_encoder, input_decoder], outputs=[mel_outputs, linear_outputs, stop_prediction, weights], name="Decoder")


    new_input_encoder = Input(shape=(None,))
    new_input_decoder = Input(shape=(None, hp.num_mels))

    encoder_out = encoder_model(new_input_encoder)
    mel_out, linear_out, stop_token_out, _ = decoder_model([encoder_out, new_input_decoder])

    model = Model(inputs=[new_input_encoder, new_input_decoder], outputs=[mel_out, linear_out, stop_token_out])

    print("\n========== Trainable model==========\n")
    model.summary(150)
    print("\n========== Encoder ==========\n")
    encoder_model.summary(150)
    print("\n========== Decoder ==========\n")
    decoder_model.summary(150)

    """ Test prediction """
    
    in1 = np.ones((1, 112))
    in2 = np.random.random((1, 256, 80))

    mel, linear, stop = model.predict([in1, in2])

    encoder_out_test = encoder_model.predict(in1)

    mel2, linear2, stop2, align = decoder_model.predict([encoder_out_test, in2])

    return model, encoder_model, deccoder_model