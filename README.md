# Tacotron2_in_keras

This is an implementation of Tacotron-2 architecture in Keras. 
It's based on the implementation (in tensorflow) of this repository : 
https://github.com/Rayhane-mamah/Tacotron-2

## TODO list

- [x] Attention mechanism (LocationSensitiveAttention) : (*, **, ***)
- [x] ZoneoutLSTM
- [x] Different blocks of Tacotron model (*)
- [x] preprocessing
- [x] training
- [x] prediction
- [ ] have results (i have results but no alignment and no result when auto-generative)
- [ ] ~~Wavenet~~ (don't have time this year...)
* these points are done and compile correctly but not sure they are correct, if anyone can confirm

** in the tensorflow implementation, the context vector is concatenated with DecoderRNN (DecoderRNN is a StackedRNNCells(...)) so, to do it, i add the cell directly into the attention_mechanism so that i can do all steps at a time (because the AttentionLayer is not a RNN instance but a Layer and uses K.rnn to iterates over decoder_timesteps and then produce all e_i and c_i). 

## Issue

- The alignment is not learned after around 5k step (perhaps not enough but with tensorflow implementation it's learned after 3.5k so...)
- The output is good when predicting with target as input but when the input isthe lasts outputs, the model can't predit anything (why ? perhaps my start of sequence (line of 0) is not a good idea ?)
- The model is very slow when predicting from it's output because i must pass all the previous outputs to produce nextoutput because i can't get internal state of attention layer (and then i can't pass it as inputto continue from lastprevious state). If anyone has an idea to improve my layer and pass states as output, he is welcome !