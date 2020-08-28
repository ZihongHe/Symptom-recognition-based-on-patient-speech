from os.path import isfile
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten
from tensorflow.keras import Model, optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Bidirectional, Concatenate, Attention, GlobalAveragePooling1D, BatchNormalization, Dropout
import time



class model:
    """
    Model to recognised prompt (label) based on the input MFCC feature
    CNN-model
    Lstm-model
    End2end-model
    Attention-model
    """
    def __init__(self, train_mfcc, train_label,  test_mfcc, test_label,  validate_mfcc, validate_label):

        self.train_mfcc = train_mfcc
        self.train_label = train_label
        self.test_mfcc = test_mfcc
        self.test_label = test_label
        self.validate_mfcc = validate_mfcc
        self.validate_label = validate_label
        self.history = None


    def CNN_model(self, learning_rate, epoch, batchsize, whether_Adam, Momentum_gamma, weight_decay, whether_load, cnn_type):
        """
        Resnet model
        :param learning_rate
        :param epoch
        :param batchsize
        :param whether_Adam: whether to perform Adam optimiser, if not perform Momentum
        :param Momentum gamma: a variable of Momentum
        :param weight_decay: weight decay for Momentum
        :param whether_load: whether to load trained Resnet model in if it exists (or cover it)
        """

        test_cnn_mfcc = self.train_mfcc
        test_cnn_label = self.train_label

        if(isfile("model/resnet_label.hdf5") and whether_load):
            self.cnn_model = load_model("model/resnet_label.hdf5")
        else:
            train_cnn_mfcc = self.test_mfcc
            train_cnn_label = self.test_label
            val_cnn_mfcc = self.validate_mfcc
            val_cnn_label = self.validate_label

            # input
            input = Input(shape=(self.test_mfcc.shape[1], self.test_mfcc.shape[2], 1))

            # Concatenate -1 dimension to be three channels, to fit the input need in ResNet50
            input_concate = Concatenate()([input,input,input])

            # CNN series network (VGG+Resnet)
            # reference: https://keras.io/api/applications/
            if(cnn_type == 'ResNet50'):
                from tensorflow.keras.applications import ResNet50
                cnn_output = ResNet50(pooling = 'avg')(input_concate)
            elif(cnn_type == 'ResNet101'):
                from tensorflow.keras.applications import ResNet101
                cnn_output = ResNet101(pooling = 'avg')(input_concate)
            elif(cnn_type == 'ResNet152'):
                from tensorflow.keras.applications import ResNet152
                cnn_output = ResNet152(pooling = 'avg')(input_concate)
            elif(cnn_type == 'ResNet50V2'):
                from tensorflow.keras.applications import ResNet50V2
                cnn_output = ResNet50V2(pooling = 'avg')(input_concate)
            elif(cnn_type == 'ResNet101V2'):
                from tensorflow.keras.applications import ResNet101V2
                cnn_output = ResNet101V2(pooling = 'avg')(input_concate)
            elif(cnn_type == 'ResNet152V2'):
                from tensorflow.keras.applications import ResNet152V2
                cnn_output = ResNet152V2(pooling = 'avg')(input_concate)
            elif(cnn_type == 'VGG16'):
                # width and height should not smaller than 32
                from tensorflow.keras.applications import VGG16
                cnn_output = VGG16(include_top = False, pooling = 'avg')(input_concate)
                cnn_output = Flatten()(cnn_output)
            elif(cnn_type == 'VGG19'):
                # width and height should not smaller than 32
                from tensorflow.keras.applications import VGG19
                cnn_output = VGG19(include_top = False, pooling = 'avg')(input_concate)
                cnn_output = Flatten()(cnn_output)
            else:
                # CNN layers we design
                print("No recognised CNN network. The CNN layers we designed are performed")
                # convolution layers
                conv_output1 = Conv2D(filters=32, strides=(1, 1), kernel_size=5, activation='relu')(input)
                # pool_output1 = MaxPool2D(pool_size=(2, 2))(conv_output1)
                conv_output2 = Conv2D(filters=8, strides=(2, 2), kernel_size=4, activation='relu')(conv_output1)

                conv_output2 = Dropout(0.2)(conv_output2)

                conv_output2_batch = BatchNormalization()(conv_output2)

                cnn_output = Flatten()(conv_output2_batch)
                cnn_output = Flatten()(cnn_output)


            # dense with sigmoid
            Dense_sigmoid = Dense(24, activation='sigmoid')(cnn_output)

            Dense_sigmoid = Dropout(0.2)(Dense_sigmoid)

            # dense output
            output = Dense(self.test_label.shape[1], activation='softmax')(Dense_sigmoid)

            # cnn model for labels recognision
            self.cnn_model = Model(input, output)

            # optimizer
            if whether_Adam:
                optimizer = optimizers.Adam(lr=learning_rate, beta_1 = Momentum_gamma, decay=weight_decay)
            else:
                optimizer = optimizers.SGD(lr=learning_rate, momentum=Momentum_gamma, nesterov=True, decay=weight_decay)
            self.cnn_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['mse', 'accuracy'])
            start = time.time()
            self.history = self.cnn_model.fit(train_cnn_mfcc, train_cnn_label, epochs=epoch, batch_size=batchsize, validation_data=[val_cnn_mfcc,val_cnn_label])
            self.training_time = time.time() - start
            self.cnn_model.save("model/resnet_label.hdf5")

        # model evaluation
        self.cnn_model.predict(test_cnn_mfcc)
        self.score = self.cnn_model.evaluate(test_cnn_mfcc, test_cnn_label)
        print("test loss: ", self.score[0], ", mse: ", self.score[1], ", accuracy", self.score[2])




    def LSTM_model(self, learning_rate, epoch, batchsize, Lstm_hidden_num, whether_Adam,
                        Momentum_gamma, weight_decay, whether_load, bidirectional = True):
        """
        LSTM with concated hidden states connected to output layer
        :param learning_rate
        :param epoch
        :param batchsize
        :param Lstm_hidden_num: number of hidden units of LSTM layer
        :param whether_Adam: whether to perform Adam optimiser, if not perform Momentum
        :param Momentum gamma: a variable of Momentum
        :param weight_decay: weight decay for Momentum
        :param whether_load: whether to load trained LSTM model in if it exists (or cover it)
        :param bidirectional: whether to use bidirectional LSTM or a normal one
        """

        test_lstm_mfcc = self.train_mfcc.reshape(self.train_mfcc.shape[0],self.train_mfcc.shape[1],self.train_mfcc.shape[2])
        test_lstm_label = self.train_label

        if (isfile("model/lstm_label.hdf5") and whether_load):
            self.lstm_model = load_model("model/lstm_label.hdf5")
        else:

            train_lstm_mfcc = self.test_mfcc.reshape(self.test_mfcc.shape[0], self.test_mfcc.shape[1],
                                                     self.test_mfcc.shape[2])
            train_lstm_label = self.test_label
            val_lstm_mfcc = self.validate_mfcc.reshape(self.validate_mfcc.shape[0], self.validate_mfcc.shape[1],
                                                     self.validate_mfcc.shape[2])
            val_lstm_label = self.validate_label

            # input
            input = Input(shape=(None, self.test_mfcc.shape[2]))
            lstm_layer = LSTM(Lstm_hidden_num, return_state=True)
            if(bidirectional):
                bi_lstm_layer = Bidirectional(lstm_layer)
                _, forword_h, forword_c, backword_h, backword_c = bi_lstm_layer(input)
                Concate = Concatenate()([forword_h, forword_c, backword_h, backword_c])
            else:
                _, h, c = lstm_layer(input)
                Concate = Concatenate()([h, c])

            Concate = Dropout(0.2)(Concate)

            Concate_batchnorm = BatchNormalization()(Concate)

            # dense output
            output = Dense(self.test_label.shape[1], activation='softmax')(Concate_batchnorm)

            # cnn model for labels recognision
            self.lstm_model = Model(input, output)

            # optimizer
            if whether_Adam:
                optimizer = optimizers.Adam(lr=learning_rate, beta_1 = Momentum_gamma, decay=weight_decay)
            else:
                optimizer = optimizers.SGD(lr=learning_rate, momentum=Momentum_gamma, nesterov=True, decay=weight_decay)

            self.lstm_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['mse', 'accuracy'])
            start = time.time()
            self.history = self.lstm_model.fit(train_lstm_mfcc, train_lstm_label, epochs=epoch, batch_size=batchsize, validation_data=[val_lstm_mfcc, val_lstm_label])
            self.training_time = time.time() - start
            self.lstm_model.save("model/lstm_label.hdf5")

        # model evaluation
        self.lstm_model.predict(test_lstm_mfcc)
        self.score = self.lstm_model.evaluate(test_lstm_mfcc, test_lstm_label)
        print("test loss: ", self.score[0], ", mse: ", self.score[1], ", accuracy", self.score[2])


    def end2end_model(self, int_to_labels, learning_rate, epoch, batchsize, Lstm_hidden_num, whether_Adam,
                        Momentum_gamma, weight_decay, whether_load):
        """
        LSTM-based end to end model
        :param int_to_labels: dictionary convert id number to prompt label
        :param learning_rate
        :param epoch
        :param batchsize
        :param Lstm_hidden_num: number of hidden units of LSTM layer
        :param whether_Adam: whether to perform Adam optimiser, if not perform Momentum
        :param Momentum gamma: a variable of Momentum
        :param weight_decay: weight decay for Momentum
        :param whether_load: whether to load trained end2end model in if it exists (or cover it)
        """


        # encoder and decoder model test data
        encoder_test = self.train_mfcc
        encoder_test = encoder_test.reshape(encoder_test.shape[0], 1, encoder_test.shape[1], encoder_test.shape[2])
        decoder_test_output = self.train_label
        decoder_size = 10
        decoder_test_input = np.zeros([1,1,decoder_size])

        if (isfile("model/end2end_encoder.hdf5") and isfile("model/end2end_decoder.hdf5") and whether_load):
            self.Model_end2end_encoder = load_model("model/end2end_encoder.hdf5")
            self.Model_end2end_decoder = load_model("model/end2end_decoder.hdf5")

        else:
            #data prerproprocess for encoder
            encoder_train = self.test_mfcc.reshape(self.test_mfcc.shape[0], self.test_mfcc.shape[1], self.test_mfcc.shape[2])
            encoder_val = self.validate_mfcc.reshape(self.validate_mfcc.shape[0], self.validate_mfcc.shape[1], self.validate_mfcc.shape[2])

            #data prerproprocess for decoder
            decoder_train_input = np.zeros([encoder_train.shape[0],1,decoder_size])
            decoder_val_input = np.zeros([encoder_val.shape[0],1,decoder_size])
            decoder_train_output = self.test_label
            decoder_val_output = self.validate_label

            # train model

            # encoder
            encoder_input = Input(shape=(None, encoder_train.shape[2]))
            encoder_output, encoder_h, encoder_c = LSTM(Lstm_hidden_num, return_state=True, return_sequences=True)(encoder_input)

            encoder_h = Dropout(0.2)(encoder_h)

            encoder_c = Dropout(0.2)(encoder_c)

            encoder_h = BatchNormalization()(encoder_h)

            encoder_c = BatchNormalization()(encoder_c)

            # decoder
            decoder_input = Input(shape=(None, decoder_train_input.shape[2]))
            decoder_lstm = LSTM(Lstm_hidden_num, return_state=True, return_sequences=True)
            decoder_output, decoder_h, decoder_c = decoder_lstm(decoder_input, initial_state=[encoder_h, encoder_c])

            decoder_output = Dropout(0.2)(decoder_output)

            decoder_dense = Dense(decoder_train_output.shape[1], activation='softmax')
            output = decoder_dense(decoder_output)

            self.Model_end2end_train = Model([encoder_input, decoder_input], output)

            # optimizer
            if whether_Adam:
                optimizer = optimizers.Adam(lr=learning_rate, beta_1 = Momentum_gamma, decay=weight_decay)
            else:
                optimizer = optimizers.SGD(lr=learning_rate, momentum=Momentum_gamma, nesterov=True, decay=weight_decay)

            self.Model_end2end_train.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['mse', 'accuracy'])

            start = time.time()
            self.history = self.Model_end2end_train.fit([encoder_train, decoder_train_input], decoder_train_output, epochs=epoch, batch_size=batchsize, validation_data=[[encoder_val, decoder_val_input], decoder_val_output])
            self.training_time = time.time() - start

            # encoder model

            self.Model_end2end_encoder = Model(encoder_input, [encoder_h, encoder_c])

            self.Model_end2end_encoder.save("model/end2end_encoder.hdf5")


            # decoder model

            decoder_h_input = Input(shape=(Lstm_hidden_num,))
            decoder_c_input = Input(shape=(Lstm_hidden_num,))

            decoder_output, decoder_h_output, decoder_c_output = decoder_lstm(decoder_input, initial_state=[decoder_h_input, decoder_c_input])

            output = decoder_dense(decoder_output)

            self.Model_end2end_decoder = Model([decoder_input, decoder_h_input, decoder_c_input], output)

            self.Model_end2end_decoder.save("model/end2end_decoder.hdf5")

        # prediction

        # test accuracy counting
        num_trueclassify = 0

        for i in range(len(encoder_test)):
            [h, c] = self.Model_end2end_encoder.predict(encoder_test[i])
            current_decoder_train_output = self.Model_end2end_decoder.predict([decoder_test_input, h, c])
            output_hat = np.argmax(current_decoder_train_output)
            output_hat = int_to_labels[output_hat]
            output = np.argmax(decoder_test_output[i])
            output = int_to_labels[output]
            if(output_hat==output):
                num_trueclassify+=1
            print("prediction:", output_hat)
            print("actually:", output,'\n')

        self.score = num_trueclassify / len(encoder_test)
        print("test accuracy:", self.score)


    def Attention_model(self, int_to_labels, learning_rate, epoch, batchsize, Lstm_hidden_num, whether_Adam,
                        Momentum_gamma, weight_decay, whether_load):
        """
        LSTM-based end to end model with global attention
        :param int_to_labels: dictionary convert id number to prompt label
        :param learning_rate:
        :param epoch:
        :param batchsize:
        :param Lstm_hidden_num: number of hidden units of LSTM layer
        :param whether_Adam: whether to perform Adam optimiser, if not perform Momentum
        :param Momentum gamma: a variable of Momentum
        :param weight_decay: weight decay for Momentum
        :param whether_load: whether to load trained Attention model in if it exists (or cover it)
        :return:
        """

        # encoder and decoder model test data
        encoder_test = self.train_mfcc
        encoder_test = encoder_test.reshape(encoder_test.shape[0], 1, encoder_test.shape[1], encoder_test.shape[2])
        decoder_test_output = self.train_label
        decoder_size = 10
        decoder_test_input = np.zeros([1,1,decoder_size])

        if (isfile("model/end2end_encoder.hdf5") and isfile("model/end2end_decoder.hdf5") and whether_load):
            self.Model_end2end_encoder = load_model("model/end2end_encoder.hdf5")
            self.Model_end2end_decoder = load_model("model/end2end_decoder.hdf5")
        else:
            #data prerproprocess for encoder
            encoder_train = self.test_mfcc.reshape(self.test_mfcc.shape[0], self.test_mfcc.shape[1], self.test_mfcc.shape[2])
            encoder_val = self.validate_mfcc.reshape(self.validate_mfcc.shape[0], self.validate_mfcc.shape[1], self.validate_mfcc.shape[2])

            #data prerproprocess for decoder
            decoder_train_input = np.zeros([encoder_train.shape[0],1,decoder_size])
            decoder_val_input = np.zeros([encoder_val.shape[0],1,decoder_size])
            decoder_train_output = self.test_label
            decoder_val_output = self.validate_label

            # train model

            # encoder
            encoder_input = Input(shape=(None, encoder_train.shape[2]))
            encoder_output, encoder_h, encoder_c = LSTM(Lstm_hidden_num, return_state=True, return_sequences=True)(encoder_input)

            encoder_h = Dropout(0.2)(encoder_h)

            encoder_c = Dropout(0.2)(encoder_c)

            encoder_output = Dropout(0.2)(encoder_output)

            encoder_h = BatchNormalization()(encoder_h)

            encoder_c = BatchNormalization()(encoder_c)

            encoder_output = BatchNormalization()(encoder_output)

            # decoder
            decoder_input = Input(shape=(None, decoder_train_input.shape[2]))
            decoder_lstm = LSTM(Lstm_hidden_num, return_state=True, return_sequences=True)
            decoder_output, decoder_h, decoder_c = decoder_lstm(decoder_input, initial_state=[encoder_h, encoder_c])

            decoder_output = Dropout(0.2)(decoder_output)

            # Global Attention layer
            attention_lstm = Attention()
            attention_distribute = attention_lstm([encoder_output, decoder_output])

            # global average pooling among attention distribution
            GlobalAveragePooling1D_attetion_layer = GlobalAveragePooling1D()
            attention_output = GlobalAveragePooling1D_attetion_layer(attention_distribute)

            # global average pooling among decoder output
            GlobalAveragePooling1D_decoder_layer = GlobalAveragePooling1D()
            attention_output_new = GlobalAveragePooling1D_decoder_layer(decoder_output)

            # Concatenate attention output and decoder output
            Concaten_lstm = Concatenate()
            Concaten_output = Concaten_lstm([attention_output_new, attention_output])

            Concaten_output = Dropout(0.2)(Concaten_output)

            decoder_dense = Dense(decoder_train_output.shape[1], activation='softmax')
            output = decoder_dense(Concaten_output)

            self.Model_Attetion_train = Model([encoder_input, decoder_input], output)

            # optimizer
            if whether_Adam:
                optimizer = optimizers.Adam(lr=learning_rate, beta_1 = Momentum_gamma, decay=weight_decay)
            else:
                optimizer = optimizers.SGD(lr=learning_rate, momentum=Momentum_gamma, nesterov=True, decay=weight_decay)

            self.Model_Attetion_train.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['mse', 'accuracy'])

            start = time.time()
            self.history = self.Model_Attetion_train.fit([encoder_train, decoder_train_input], decoder_train_output, epochs=epoch, batch_size=batchsize, validation_data=[[encoder_val, decoder_val_input], decoder_val_output])
            self.training_time = time.time() - start

            # encoder model

            self.Model_Attetion_encoder = Model(encoder_input, [encoder_output, encoder_h, encoder_c])

            self.Model_Attetion_encoder.save("model/Attention_encoder.hdf5")

            # decoder model

            decoder_h_input = Input(shape=(Lstm_hidden_num,))
            decoder_c_input = Input(shape=(Lstm_hidden_num,))
            encoder_output_for_attention = Input(shape = (None, Lstm_hidden_num))

            decoder_output, decoder_h_output, decoder_c_output = decoder_lstm(decoder_input, initial_state=[decoder_h_input, decoder_c_input])

            attention_distribute = attention_lstm([encoder_output_for_attention, decoder_output])

            attention_output = GlobalAveragePooling1D_attetion_layer(attention_distribute)

            decoder_output_new = GlobalAveragePooling1D_decoder_layer(decoder_output)

            Concaten_output = Concaten_lstm([decoder_output_new, attention_output])

            output = decoder_dense(Concaten_output)

            self.Model_Attetion_decoder = Model([decoder_input, decoder_h_input, decoder_c_input, encoder_output_for_attention], output)

            self.Model_Attetion_decoder.save( "model/Attention_decoder.hdf5")

        # prediction

        # test accuracy counting
        num_trueclassify = 0

        for i in range(len(encoder_test)):
            [encoder_out, h, c] = self.Model_Attetion_encoder.predict(encoder_test[i])
            current_decoder_train_output = self.Model_Attetion_decoder.predict([decoder_test_input, h, c, encoder_out])
            output_hat = np.argmax(current_decoder_train_output)
            output_hat = int_to_labels[output_hat]
            output = np.argmax(decoder_test_output[i])
            output = int_to_labels[output]
            if(output_hat==output):
                num_trueclassify+=1
            #print("prediction:", output_hat)
            #print("actually:", output,'\n')

        self.score = num_trueclassify / len(encoder_test)
        print("test accuracy:", self.score)