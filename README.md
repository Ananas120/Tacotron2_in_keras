# Tacotron2_in_keras

This is an implementation of Tacotron-2 architecture in Keras. 
It's based on the implementation (in tensorflow) of this repository : 
https://github.com/Rayhane-mamah/Tacotron-2
The implementation is not not done yet : 
- [x] Attention mechanism (LocationSensitiveAttention) : (*, **, ***)
- [x] ZoneoutLSTM : (*)
- [x] Different blocks of Tacotron model : (*)
- [x] preprocessing
- [x] training
- [x] prediction
- [ ] have results
- [ ] ~~Wavenet~~ (it seems too difficult in Keras)
* these points are done and compile correctly but not sure they are correct, if anyone can confirm

** in the tensorflow implementation, the context vector is concatenated with DecoderRNN (DecoderRNN is a StackedRNNCells(...)) so, to do it, i add the cell directly into the attention_mechanism so that i can do all steps at a time (because the AttentionLayer is not a RNN instance but a Layer and uses K.rnn to iterates over decoder_timesteps and then produce all e_i and c_i). 

*** strange thing when i call attention_layer.__call__(...) with only 'decoder_outputs' as input, it does exception but when i call with [encoder_outputs, decoder_outputs] it compiles even though i never use 'encoder_outputs' in the method. 

## Issue

The code runs (i will post the main object and training code tomorrow) but when i train the model, the weights becomes 'nan' after around 40-50 steps and i don't understand why... During this 40 first steps, the loss goes down and after it goes (in 3-4 steps) from ~10 to ~40 (which is themaximum because of output clipping). If anyone has an idea... 

- I sort the batches by number of mel frames so batch 40+ are around 250-300 frames (with outputs_per_step = 2) (according to my dataset) so perhaps gradients exploded when timestep is too high ? 
- Perhaps my ZoneoutLSTM is not well implemented and cause errors ? or my attention mechanism ?
