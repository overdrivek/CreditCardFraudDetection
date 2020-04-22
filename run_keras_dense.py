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

parser = argparse.ArgumentParser()
parser.add_argument('--train_file',type=str,default='.\\data\\smote_oversampled\\Training_data.csv')
parser.add_argument('--test_file',type=str,default='.\\data\\Test_data.csv')
parser.add_argument('--validation_file',type=str,default='.\\data\\Validation_data.csv')
parser.add_argument('--target_class',type=str,default='Class')
parser.add_argument('--optimizer',type=str,default='SGD')
parser.add_argument('--lr',type=float,default=0.01)
parser.add_argument('--momentum',type=float,default=0.8)
parser.add_argument('--loss',type=str,default='binary_crossentropy')
parser.add_argument('--epochs',type=int,default=50)
parser.add_argument('--batch_size',type=int,default=256)

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


        self.X_train = self.df_train_file.drop([self.target_label],axis=1)
        self.Y_train = self.df_train_file[self.target_label]

        self.X_validation = self.df_validation_file.drop([self.target_label], axis=1)
        self.Y_validation = self.df_validation_file[self.target_label]

        self.X_test = self.df_test_file.drop([self.target_label], axis=1)
        self.Y_test = self.df_test_file[self.target_label]

        self.init_model()
        self.init_optimizer()
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy', k_metrics.f1_m])

        self.train(self.X_train,self.Y_train)
        print('Validation results')
        validation_metrics = self.evaluate(self.X_validation,self.Y_validation)

        print('Test results')
        test_metrics = self.evaluate(self.X_test, self.Y_test)


    def train(self,X_train,Y_train):
        self.model.fit(X_train,Y_train,epochs=self.epochs,batch_size=self.batch_size,validation_data=(self.X_validation,self.Y_validation),
                       verbose=2,class_weight={0: 1, 1: 2})
        feature_names = np.asarray(self.df_train_file.drop(['Class'], axis=1).columns)

    def plot_roc(self,fpr, tpr, auc, model_name):
        plt.plot(fpr, tpr, label=model_name + ' (AUC = ' + str(auc) + ')')
        plt.legend()
        plt.grid(True)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.show()

    def evaluate(self,X_validation,Y_validation):
        performance_metrics = {}
        if X_validation is not None and Y_validation is not None:
            prediction = self.model.predict(X_validation)
            fpr, tpr, threshold = metrics.roc_curve(Y_validation, prediction, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            self.plot_roc(fpr, tpr, auc, 'SVC')

            prediction = self.model.predict_classes(X_validation)
            confusion_matrix = metrics.confusion_matrix(Y_validation, prediction)
            accuracy = metrics.accuracy_score(Y_validation, prediction)


            print('Area under the curve = ', auc)
            print('Confusion matrix \n ', confusion_matrix)
            print('Accuracy = ', accuracy)
            print('Classification metrics = \n', metrics.classification_report(Y_validation, prediction))

            performance_metrics['fpr'] = fpr
            performance_metrics['tpr'] = tpr
            performance_metrics['auc'] = auc
            performance_metrics['confusion_matrix'] = confusion_matrix
            performance_metrics['accuracy'] = accuracy
            return performance_metrics

    def init_model(self):

        self.model = Sequential()
        self.model.add(Dense(64, input_dim=self.X_train.shape[1]))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dense(16))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dense(1, activation='sigmoid'))

    def init_optimizer(self):
        if self.optimizer_type == 'Adam':
            self.optimizer = Adam(learning_rate=self.lr)
        else:
            self.optimizer = SGD(learning_rate=self.lr,momentum=self.momentum)


if __name__=='__main__':
    args = parser.parse_args()
    classifier = Dense_Classifier(args)
    classifier.run()
