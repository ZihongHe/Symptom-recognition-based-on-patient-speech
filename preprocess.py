from python_speech_features import mfcc
import numpy as np
from tensorflow.keras import preprocessing
from tensorflow.keras.utils import to_categorical
from collections import Counter
import librosa

class preprocess:
    """
    This class is for audio preprocess
    """

    def raw(self,  wav_path, rate0_sig1, coefficient_samplerate, numcep, nfilt):
        """
        Package of mfcc extracted see: https://python-speech-features.readthedocs.io/en/latest/
        :param sr: sample rate used to load audio files
        :param wav_path: wav audio path
        :param rate0_sig1: whether to choose rate or length of signal as MFCC's sample_rate
                           (if 0: i.e. using rate, revert original sample rate of audio;
                            if 1: i.e. using length of signal, make all MFCC shape the same without padding)
        :param coefficient_samplerate: coefficient of sample rate
        :param numcep: number of cepstrum to return
        :param nfilt: number of filters in the filterbank
        :return: MFCC: extracted mfcc features, a numpy array of size (NUMFRAMES by numcep) containing features.
                       Each row holds 1 feature vector.
        """

        # load audio files using desired sample rate (in Hz)
        #sig, rate = librosa.load(wav_path, sr=sr)
        sig, rate = librosa.load(wav_path, sr=None)

        if (rate0_sig1):
            samplerate = len(sig) * coefficient_samplerate
        else:
            samplerate = rate * coefficient_samplerate

        nfft = int(samplerate * 0.025) + 1  # 7394  # maximum number of points in one frame among all audio files: max(len(sig))*winlen
        # the default window length of 25ms and window step of 10ms are applied
        MFCC = mfcc(sig, samplerate=samplerate, numcep=numcep, nfilt=nfilt, nfft=nfft, highfreq=min(8000, samplerate/2))
        #MFCC_T = librosa.feature.mfcc(y=sig, sr=samplerate, n_mfcc=numcep, lifter=22, n_fft=nfft, hop_length=int(samplerate*0.01), fmax=min(8000, samplerate/2), htk=False)
        #MFCC = np.swapaxes(MFCC, 0, 1)

        #melspec = librosa.feature.melspectrogram(y=sig, sr=samplerate, n_fft=nfft, n_mels=nfilt, hop_length=int(samplerate * 0.01), fmax=min(8000, samplerate/2))
        #powerspec_T = librosa.power_to_db(melspec, ref=np.max)
        #powerspec = np.swapaxes(powerspec, 0, 1)

        return MFCC

    def mfcc_padding(self, train_mfcc, test_mfcc, validate_mfcc):
        """
        This function is to padding mfcc features with 0 if they have different shape
        :param train_mfcc: train samples
        :param test_mfcc:  test samples
        :param validate_mfcc: validate samples
        :return: train_mfcc, test_mfcc, validate_mfcc: after-padding samples
        """

        num_train = train_mfcc.shape[0]

        num_test = test_mfcc.shape[0]

        num_validate = validate_mfcc.shape[0]

        MFCC = np.concatenate((train_mfcc, test_mfcc, validate_mfcc))

        originalshape = MFCC.shape

        MFCC = preprocessing.sequence.pad_sequences(MFCC, maxlen=None, dtype='float', padding='post', truncating='post', value=0)

        newshape = MFCC.shape

        if(originalshape == newshape):
            print("Each mfcc has the same shape, no padding need.")
        else:
            train_mfcc = MFCC[:num_train]
            test_mfcc = MFCC[num_train:num_train+num_test]
            validate_mfcc = MFCC[num_train+num_test:]

        return train_mfcc, test_mfcc, validate_mfcc


    def labels_dictionary(self, train_label, test_label, validate_label):
        """
        This is to create dictionary for labels
        :param train_label: original string-type train label
        :param test_label: original string-type test label
        :param validate_label: original string-type validate label
        """

        label = sorted(list(set((train_label+test_label+validate_label))))
        self.int_to_labels = dict((i,labels) for i, labels in enumerate(label))
        self.labels_to_int = dict((labels, i) for i, labels in enumerate(label))

    def labels(self, train_label, test_label, validate_label):
        """
        Convert labels into one-hot style
        :param train_label: original string-type train label
        :param test_label: original string-type test label
        :param validate_label: original string-type validate label
        :return: train_label, test_label, validate_label: one-hot style labels
        """

        # Counting distribution of each prompt in different data set
        self.train_counter = Counter(train_label)
        self.test_counter = Counter(test_label)
        self.validate_counter = Counter(validate_label)

        print("prompt's distribution for train(actually test) data:", self.train_counter)
        print("prompt's distribution for test(actually train) data:", self.test_counter)
        print("prompt's distribution for validate data:", self.validate_counter)

        # dictionary for labels & integer exchanging
        self.labels_dictionary(train_label, test_label, validate_label)

        train_label = [self.labels_to_int[i] for i in train_label]
        print("train label finish preprocess")
        test_label = [self.labels_to_int[i] for i in test_label]
        print("test label finish preprocess")
        validate_label = [self.labels_to_int[i] for i in validate_label]
        print("validate label finish preprocess")

        # number to categorical
        train_label = to_categorical(train_label, num_classes=len(self.labels_to_int))
        test_label = to_categorical(test_label, num_classes=len(self.labels_to_int))
        validate_label = to_categorical(validate_label, num_classes=len(self.labels_to_int))

        return train_label, test_label, validate_label



