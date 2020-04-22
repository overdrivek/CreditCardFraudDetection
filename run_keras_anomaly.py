import argparse
import pandas as pd
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
import keras_metrics as k_metrics
from sklearn.metrics import precision_recall_curve

parser = argparse.ArgumentParser()
parser.add_argument('--train_file',type=str,default='.\\data\\random_oversampled\\Training_data.csv')
parser.add_argument('--test_file',type=str,default='.\\data\\Test_data.csv')
parser.add_argument('--validation_file',type=str,default='.\\data\\Validation_data.csv')
parser.add_argument('--target_class',type=str,default='Class')
parser.add_argument('--optimizer',type=str,default='Adam')
parser.add_argument('--lr',type=float,default=0.0001)
parser.add_argument('--momentum',type=float,default=0.8)
parser.add_argument('--loss',type=str,default='mse')
parser.add_argument('--epochs',type=int,default=50)
parser.add_argument('--batch_size',type=int,default=512)

class Dense_Classifier:

    def __init__(self,args):
        self.train_file = args.train_file
        self.validation_file = args.validation_file
        self.test_file = args.test_file
        self.target_label = args.target_class
        self.optimizer_type = args.optimizer
        self.lr = args.lr
        self.momentum = args.momentum
        self.loss = args.loss
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.read_files()

    def read_files(self):
        self.df_train_file = pd.read_csv(self.train_file)
        self.df_validation_file = pd.read_csv(self.validation_file)
        self.df_test_file = pd.read_csv(self.test_file)

    def run(self):


        X_train = self.df_train_file.drop([self.target_label],axis=1)
        Y_train = self.df_train_file[self.target_label]
        self.X_train = X_train[Y_train==0]


        self.X_validation = self.df_validation_file.drop([self.target_label], axis=1)
        self.Y_validation = self.df_validation_file[self.target_label]

        X_test = self.df_test_file.drop([self.target_label], axis=1)
        Y_test = self.df_test_file[self.target_label]
        self.X_test = X_test[Y_test == 0]

        self.init_model()
        self.init_optimizer()
        self.model.compile(loss=self.loss, optimizer=self.optimizer)

        self.train(self.X_train,self.X_validation)
        print('Validation results')
        validation_metrics = self.evaluate(self.X_validation,self.Y_validation)

        print('Test results')
        test_metrics = self.evaluate(X_test, Y_test)


    def train(self,X_train,X_validation):
        self.model.fit(X_train,X_train,epochs=self.epochs,batch_size=self.batch_size,validation_data=(X_validation,X_validation),
                       verbose=2)
        feature_names = np.asarray(self.df_train_file.drop(['Class'], axis=1).columns)

    def plot_roc(self,fpr, tpr, auc, model_name):
        plt.plot(fpr, tpr, label=model_name + ' (AUC = ' + str(auc) + ')')
        plt.legend()
        plt.grid(True)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.show()

    def evaluate(self,X_validation,Y_validation):
        predicted_output = self.model.predict(X_validation)
        # predicted_output = autoencoder.predict(X_train)
        mse = np.mean(np.power(X_validation- predicted_output, 2), axis=1)
        # norm=mse/(max(mse)-min(mse))
        norm = mse
        error_df = pd.DataFrame({'Reconstruction_error': norm,
                                 'True_class': Y_validation})

        precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class,
                                                                       error_df.Reconstruction_error)
        plt.plot(threshold_rt, precision_rt[1:], label="Precision", linewidth=5)
        plt.plot(threshold_rt, recall_rt[1:], label="Recall", linewidth=5)
        plt.title('Precision and recall for different threshold values')
        plt.xlabel('Threshold')
        plt.ylabel('Precision/Recall')
        plt.legend()
        plt.show()

        threshold = 10
        y_predicted = np.array(norm > threshold).astype(np.int)
        fpr, tpr, threshold = metrics.roc_curve(Y_validation, y_predicted, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        self.plot_roc(fpr, tpr, auc, 'anomaly')
        confusion_matrix = metrics.confusion_matrix(Y_validation, y_predicted)
        print(confusion_matrix)
        accuracy = metrics.accuracy_score(Y_validation, y_predicted)
        print(accuracy)
        print('Classification metrics = \n', metrics.classification_report(Y_validation, y_predicted))

    def init_model(self):

        self.model = Sequential()
        encoder_anomaly = Sequential()
        encoder_anomaly.add(Dense(64, input_dim=self.X_train.shape[1]))
        encoder_anomaly.add(BatchNormalization())
        encoder_anomaly.add(Activation('relu'))
        encoder_anomaly.add(Dense(32))
        encoder_anomaly.add(BatchNormalization())
        encoder_anomaly.add(Activation('relu'))
        encoder_anomaly.add(Dense(16, activation='relu'))
        encoder_anomaly.summary()
        # inputs = Input(shape=(X_resampled.shape[1],),name='decoder_input')
        # latent_inputs = Input(shape=(1,),name='decoder_input')

        # decoder_anomaly = Sequential()
        decoder_anomaly = Sequential()
        decoder_anomaly.add(Dense(32, input_dim=16))
        decoder_anomaly.add(BatchNormalization())
        decoder_anomaly.add(Activation('relu'))
        decoder_anomaly.add(Dense(64))
        decoder_anomaly.add(BatchNormalization())
        decoder_anomaly.add(Activation('relu'))
        decoder_anomaly.add(Dense(self.X_train.shape[1], activation='sigmoid'))
        decoder_anomaly.summary()
        self.model.add(encoder_anomaly)
        self.model.add(decoder_anomaly)
        self.model.summary()

    def init_optimizer(self):
        if self.optimizer_type == 'Adam':
            self.optimizer = Adam(learning_rate=self.lr)
        else:
            self.optimizer = SGD(learning_rate=self.lr,momentum=self.momentum)


if __name__=='__main__':
    args = parser.parse_args()
    classifier = Dense_Classifier(args)
    classifier.run()
