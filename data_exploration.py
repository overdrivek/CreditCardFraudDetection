import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from imblearn.over_sampling import SMOTE,RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import os

class DataExploration:
    def __init__(self,input_file=None):
        self.input_file = input_file
        self.df_input = None
        self.col_names = None
        self.target_column = 'Class'
        self.test_split = 0.3
        self.export_folder = '.\\data\\'

    def run(self):
        input_data = pd.read_csv(self.input_file)
        self.credit_card_get_data_insights(input_data)
        self.df_input = self.clean_data(input_data)
        self.get_training_test_data()
        self.get_undersampled_data()

    def credit_card_get_data_insights(self,input_df):
        if input_df is not None:
            print('Number of samples = ', input_df.shape[0])
            classes = pd.unique(input_df[self.target_column])
            print('Number of classes = ', len(classes))
            print('Number of features = ', input_df.shape[1])
            class_groups = input_df.groupby([self.target_column])
            print('Number of samples per class')
            print(class_groups.apply(len))

    def clean_data(self,input_data):
        print('Number of NaN values in the data = ', input_data.isna().sum().sum())
        df_input_cleaned = input_data.fillna(0)

        credit_card_fd_input_standardized = self.scale_data(df_input_cleaned)
        return credit_card_fd_input_standardized

    def scale_data(self,df_input_cleaned):
        std_scaler = preprocessing.StandardScaler()
        std_scaler.fit(df_input_cleaned.loc[:, ['Time', 'Amount']])
        data_transformed = std_scaler.transform(df_input_cleaned.loc[:, ['Time', 'Amount']])
        credit_card_fd_input_standardized = df_input_cleaned.drop(['Time', 'Amount'], axis=1)
        credit_card_fd_input_standardized['scaledTime'] = data_transformed[:, 0]
        credit_card_fd_input_standardized['scaledAmount'] = data_transformed[:, 1]
        return credit_card_fd_input_standardized

    def get_training_test_data(self):
        self.col_names = self.df_input.drop(['Class'],axis=1).columns
        input_data = np.asarray(self.df_input.drop(['Class'],axis=1))
        target_data = np.asarray(self.df_input['Class'])
        X_train_valid, self.X_test, Y_train_valid, self.Y_test = train_test_split(input_data,target_data, test_size=0.1, random_state=42)
        self.X_train, self.X_validation, self.Y_train, self.Y_validation = train_test_split(X_train_valid, Y_train_valid,test_size=0.2, random_state=42)
        print('Training')
        self.class_count_fnc(self.Y_train)
        print('Validation')
        self.class_count_fnc(self.Y_validation)
        print('Test')
        self.class_count_fnc(self.Y_test)

        df_train_data = pd.DataFrame(self.X_train,columns=self.col_names)
        df_train_data['Class']=self.Y_train
        df_train_data.to_csv('.\\data\\Training_data.csv',index=None)

        df_validation_data = pd.DataFrame(self.X_validation, columns=self.col_names)
        df_validation_data['Class'] = self.Y_validation
        df_validation_data.to_csv('.\\data\\Validation_data.csv', index=None)

        df_test_data = pd.DataFrame(self.X_test, columns=self.col_names)
        df_test_data['Class'] = self.Y_test
        df_test_data.to_csv('.\\data\\Test_data.csv',index=None)


    def class_count_fnc(self,label_array=None,fp=None):
        for class_ in np.unique(label_array):
            if fp is not None:
                fp.write('Class {} : Number of samples = {} \n'.format(class_, len(np.where(label_array == class_)[0])))
            print('Class {} : Number of samples = {}'.format(class_, len(np.where(label_array == class_)[0])))

    def get_undersampled_data(self):
        # Undersampling
        under_sampler = RandomUnderSampler(random_state=42,replacement=False)
        X_undersampled, Y_undersampled = under_sampler.fit_resample(self.X_train,self.Y_train)
        print('Random under sampling done.')
        self.export_files(base_folder=self.export_folder,folder_name='random_undersampled',X_input=X_undersampled,Y_input=Y_undersampled)


        # Oversampling
        over_sampler = RandomOverSampler(random_state=42)
        X_oversampled, Y_oversampled = over_sampler .fit_resample(self.X_train, self.Y_train)
        print('Random over sampling done.')
        self.export_files(base_folder=self.export_folder, folder_name='random_oversampled', X_input=X_oversampled,
                          Y_input=Y_oversampled)


        # SMOTE Oversampling
        smote_over_sampler = SMOTE(random_state=42)
        X_oversampled, Y_oversampled = smote_over_sampler.fit_resample(self.X_train, self.Y_train)
        print('SMOTE over sampling done.')
        self.export_files(base_folder=self.export_folder, folder_name='smote_oversampled', X_input=X_oversampled,
                          Y_input=Y_oversampled)

    def export_files(self,base_folder=None,folder_name='',X_input=None,Y_input=None):
        folder_export = os.path.join(base_folder, folder_name)
        if os.path.exists(folder_export) is False:
            os.mkdir(folder_export)
        df_train_data = pd.DataFrame(X_input, columns=self.col_names)
        df_train_data['Class'] = Y_input
        export_file = os.path.join(folder_export, 'Training_data.csv')
        df_train_data.to_csv(export_file, index=None)
        with open(os.path.join(folder_export, 'Readme.txt'), 'w') as fp:
            self.class_count_fnc(Y_input,fp)
        fp.close()

if __name__=='__main__':
    input_file = '.\\data\\creditcard.csv'
    explorer = DataExploration(input_file)
    explorer.run()
