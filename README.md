This is a symptom recognition system based on MFCC features from patient self-description wav audio

Dataset: Medical speech, transcription, and intent, url: https://www.figure-eight.com/dataset/audio-recording-and-transcription-for-medical-scenarios/, March 2020.
Please put all dataset files in a package. Then create a package name "train" inside and put train wav in; Then create a package name "test" inside and put test wav in; Then create a package name "validate" inside and put validate wav in;

Tensorflow requirement: tensorflow==2.1.0

CNN/LSTM/LSTM-based End to end/ Attention model are used in this project.

Configuration csv file is to load in parameters. Keep it at the same load path code

You need to create two package -- "preprocess", "model" and "record" at the same path with code

Run main.py, and the system will generate hdf5 model, preprocessing data and record data(include model history, test score, model building time and configuration)


Configuration:

data_path: your data set package path 

whether_load_preprocess: whether to load preprocessed data if it exists (or cover it)

rate0_sig1: whether to choose rate or length of signal as MFCC's sample_rate
            (if rate: revert original sample rate of audio; if length of signal: make all MFCC shape the same without padding)
                           
coefficient_samplerate: coeffient of sample rate

numcep: number of cepstrum to return

nfilt: number of filters in the filterbank

model_choose: which model to perform

Lstm_hidden_num: number of hidden units of LSTM layer

bidirectional: whether lstm model's lstm modules bidirectional or not

whether_Adam: whether to perform Adam optimiser, if not perform Momentum

Momentum gamma: parameter of Momentum

weight_decay: weight decay for Momentum

whether_load_model: whether to load trained model if it exists (or cover it)

record_name: the path you would like to save the generated record docs
