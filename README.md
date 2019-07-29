# Tacotron2_in_keras

This is an implementation of Tacotron-2 architecture in Keras. 
It's based on the implementation (in tensorflow) of this repository : 
https://github.com/Rayhane-mamah/Tacotron-2
The implementation is not not done yet : 
[x] Attention mechanism (LocationSensitiveAttention) : Done (*) (**) (***)
[x] ZoneoutLSTM : Done (*)
[x] Different blocks of Tacotron model : Done (*)
[x] preprocessing
[_] training
[_] prediction
[_] Wavenet (it seems too difficult in Keras)
* these points are done and compile correctly but not sure they are correct, if anyone can confirm

** in the tensorflow implementation, the context vector is concatenated with DecoderRNN (DecoderRNN is a StackedRNNCells(...)) so, to do it, i add the cell directly into the attention_mechanism so that i can do all steps at a time (because the AttentionLayer is not a RNN instance but a Layer and uses K.rnn to iterates over decoder_timesteps and then produce all e_i and c_i). 

*** strange thing when i call attention_layer.__call__(...) with only 'decoder_outputs' as input, it does exception but when i call with [encoder_outputs, decoder_outputs] it compiles even though i never use 'encoder_outputs' in the method. 
