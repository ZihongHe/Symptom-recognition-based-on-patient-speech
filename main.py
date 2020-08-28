
from os.path import isfile
import csv
import pandas as pd
import pickle
import numpy as np
from preprocess import preprocess
from model import model


if __name__ == '__main__':
    """
    This project is to recognise prompt based on light-noise extracted MFCC features from patient self-description
    Model we use: CNN/LSTM/LSTM-based end to end/Attention model
    """

    # load in configuration from command line
    configuration = list(csv.reader(open("configuration.csv")))

    path = configuration[1][0]
    audio = preprocess()
    preprocess_load = "preprocess/"

    # whether to load preprocessed data if it exists (or cover it)
    whether_load_preprocess = int(configuration[1][1])

    # notice, test set is much bigger than train set allocated by original data set, which is a bad allocation, so exchange them
    if (isfile(preprocess_load+"label_dic.txt") and whether_load_preprocess):

        preprocess_read = open(preprocess_load + "train.txt", "rb")
        (train_mfcc, train_label) = pickle.load(preprocess_read)
        preprocess_read.close()
        preprocess_read = open(preprocess_load + "test.txt", "rb")
        (test_mfcc, test_label) = pickle.load(preprocess_read)
        preprocess_read.close()
        preprocess_read = open(preprocess_load + "validate.txt", "rb")
        (validate_mfcc, validate_label) = pickle.load(preprocess_read)
        preprocess_read.close()
        preprocess_read = open(preprocess_load + "label_dic.txt", "rb")
        (audio.int_to_labels, audio.labels_to_int) = pickle.load(preprocess_read)
        preprocess_read.close()
    else:
        # initialization
        train_mfcc = []
        train_label = []
        test_mfcc = []
        test_label = []
        validate_mfcc = []
        validate_label = []

        record_frame = pd.read_csv(path + "recordings-overview.csv")
        record_filename = record_frame['file_name']
        record_prompt = record_frame['prompt']

        for i in range(len(record_frame)):

            # audio preprocessing and save into train-test-validate
            if (isfile(path + "train/" + record_filename[i])):
                MFCC = audio.raw(wav_path = path + "train/" + record_filename[i], rate0_sig1 = int(configuration[1][2]), coefficient_samplerate = float(configuration[1][3]), numcep = int(configuration[1][4]), nfilt = int(configuration[1][5]))
                train_mfcc.append(MFCC)
                train_label.append(record_prompt[i])

            if (isfile(path + "test/" + record_filename[i])):
                MFCC = audio.raw(wav_path = path + "test/" + record_filename[i], rate0_sig1 = int(configuration[1][2]), coefficient_samplerate = float(configuration[1][3]), numcep = int(configuration[1][4]), nfilt = int(configuration[1][5]))
                test_mfcc.append(MFCC)
                test_label.append(record_prompt[i])

            if (isfile(path + "validate/" + record_filename[i])):
                MFCC = audio.raw(wav_path = path + "validate/" + record_filename[i], rate0_sig1 = int(configuration[1][2]), coefficient_samplerate = float(configuration[1][3]), numcep = int(configuration[1][4]), nfilt = int(configuration[1][5]))
                validate_mfcc.append(MFCC)
                validate_label.append(record_prompt[i])

            print("Audio been processed: ", i + 1, "/", len(record_frame))

        train_mfcc = np.asarray(train_mfcc)
        test_mfcc = np.asarray(test_mfcc)
        validate_mfcc = np.asarray(validate_mfcc)

        # convert label into one-hot format
        train_label, test_label, validate_label = audio.labels(train_label, test_label, validate_label)

        #Save in preprocess data file
        preprocess_save = open(preprocess_load+"train.txt", "wb")
        pickle.dump((train_mfcc, train_label), preprocess_save)
        preprocess_save.close()
        preprocess_save = open(preprocess_load+"test.txt", "wb")
        pickle.dump((test_mfcc, test_label), preprocess_save)
        preprocess_save.close()
        preprocess_save = open(preprocess_load+"validate.txt", "wb")
        pickle.dump((validate_mfcc, validate_label), preprocess_save)
        preprocess_save.close()
        preprocess_save = open(preprocess_load+"label_dic.txt", "wb")
        pickle.dump((audio.int_to_labels, audio.labels_to_int), preprocess_save)
        preprocess_save.close()

    train_mfcc, test_mfcc, validate_mfcc = audio.mfcc_padding(train_mfcc, test_mfcc, validate_mfcc)

    train_mfcc = train_mfcc.reshape(train_mfcc.shape[0], train_mfcc.shape[1], train_mfcc.shape[2], 1)
    test_mfcc = test_mfcc.reshape(test_mfcc.shape[0], test_mfcc.shape[1], test_mfcc.shape[2], 1)
    validate_mfcc = validate_mfcc.reshape(validate_mfcc.shape[0], validate_mfcc.shape[1], validate_mfcc.shape[2], 1)

    MODEL = model(train_mfcc, train_label, test_mfcc, test_label, validate_mfcc, validate_label)

    #model related parameter setting
    model_choose = configuration[3][0]
    learning_rate = float(configuration[3][1])
    epoch = int(configuration[3][2])
    batch_size = int(configuration[3][3])
    Lstm_hidden_num = int(configuration[3][4])
    bidirectional = int(configuration[3][5])
    whether_Adam = int(configuration[3][6])
    Momentum_gamma = float(configuration[3][7])
    weight_decay = float(configuration[3][8])
    whether_load_model = int(configuration[3][9])
    cnn_type = configuration[3][10]

    # model choosing for prompt recognition
    if (model_choose == "cnn"):
        MODEL.CNN_model(learning_rate=learning_rate, epoch=epoch, batchsize=batch_size, whether_Adam = whether_Adam,
                        Momentum_gamma = Momentum_gamma, weight_decay = weight_decay, whether_load = whether_load_model,
                        cnn_type = cnn_type)
    elif (model_choose == 'lstm'):
        MODEL.LSTM_model(learning_rate=learning_rate, epoch=epoch, batchsize=batch_size, Lstm_hidden_num=Lstm_hidden_num,
                         bidirectional=bidirectional, whether_Adam = whether_Adam, Momentum_gamma = Momentum_gamma,
                         weight_decay = weight_decay,whether_load = whether_load_model)
    elif (model_choose == 'end2end'):
        MODEL.end2end_model(int_to_labels=audio.int_to_labels, learning_rate=learning_rate, epoch=epoch,
                            batchsize=batch_size, whether_Adam = whether_Adam, Momentum_gamma = Momentum_gamma,
                            weight_decay = weight_decay, Lstm_hidden_num=Lstm_hidden_num,
                            whether_load = whether_load_model)
    elif (model_choose == 'attention'):
        MODEL.Attention_model(int_to_labels=audio.int_to_labels, learning_rate=learning_rate, epoch=epoch,
                              batchsize=batch_size, whether_Adam = whether_Adam, Momentum_gamma = Momentum_gamma,
                              weight_decay = weight_decay, Lstm_hidden_num=Lstm_hidden_num,
                              whether_load = whether_load_model)
    else:
        print("model type cannot recognise, please type cnn / lstm / end2end / attention.")

    # record model history, test score, model building time and configuration into a file
    if(MODEL.history):
        record_name = configuration[5][0]
        record = open("record/" + record_name, "wb")
        pickle.dump((MODEL.history.history, MODEL.score, MODEL.training_time, configuration), record)
        record.close()



