import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--train_file',type=str,default='.\\data\\random_oversampled\\Training_data.csv')
parser.add_argument('--test_file',type=str,default='.\\data\\Test_data.csv')
parser.add_argument('--validation_file',type=str,default='.\\data\\Validation_data.csv')
parser.add_argument('--target_class',type=str,default='Class')
class RF_Classifier:

    def __init__(self,args):
        self.train_file = args.train_file
        self.validation_file = args.validation_file
        self.test_file = args.test_file
        self.target_label = args.target_class
        self.read_files()

    def read_files(self):
        self.df_train_file = pd.read_csv(self.train_file)
        self.df_validation_file = pd.read_csv(self.validation_file)
        self.df_test_file = pd.read_csv(self.test_file)

    def run(self):
        self.init_model()
        X_train = self.df_train_file.drop([self.target_label],axis=1)
        Y_train = self.df_train_file[self.target_label]

        X_validation = self.df_validation_file.drop([self.target_label], axis=1)
        Y_validation = self.df_validation_file[self.target_label]

        X_test = self.df_test_file.drop([self.target_label], axis=1)
        Y_test = self.df_test_file[self.target_label]

        self.train(X_train,Y_train)
        print('Validation results')
        validation_metrics = self.evaluate(X_validation,Y_validation)

        print('Test results')
        test_metrics = self.evaluate(X_test, Y_test)


    def train(self,X_train,Y_train):
        self.selector.fit(X_train,Y_train)
        feature_names = np.asarray(self.df_train_file.drop(['Class'], axis=1).columns)
        self.print_feature_importance(feature_names)


    def print_feature_importance(self,keys):
        fig, ax = plt.subplots(figsize=(16, 6))
        plt.plot(self.selector.feature_importances_)
        plt.xticks(range(0, len(keys)))
        ax.set_xticklabels(keys)
        plt.xlabel('Features')
        plt.ylabel('Feature importance')
        plt.grid(True)
        plt.show()

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
            prediction = self.selector.predict_proba(X_validation)
            fpr, tpr, threshold = metrics.roc_curve(Y_validation, prediction[:, 1], pos_label=1)
            auc = metrics.auc(fpr, tpr)
            self.plot_roc(fpr, tpr, auc, 'SVC')

            prediction = self.selector.predict(X_validation)
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
        self.selector = RandomForestClassifier(max_depth=None,oob_score=True,random_state=42,verbose=1,n_jobs=-1,
                                               class_weight={0: 1, 1: 2})


if __name__=='__main__':
    args = parser.parse_args()
    classifier = RF_Classifier(args)
    classifier.run()
