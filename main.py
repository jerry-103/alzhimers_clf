### Main file
#Imports
from loaders import csv_to_df
from clean import clean, split_data
from train import train_pipe
from eval import evaluate

#list of cols to drop while cleaning
cols_to_drop = ['PatientID', 'DoctorInCharge']
#list of categorical features for one-hot encoding
categorical_feats = ['Gender', 'Ethnicity', 'EducationLevel', 'Smoking',
                     'FamilyHistoryAlzheimers', 'CardiovascularDisease',
                     'Diabetes', 'Depression', 'HeadInjury', 'Hypertension',
                     'MemoryComplaints', 'BehavioralProblems', 'Confusion',
                     'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks',
                     'Forgetfulness']
#list of numerical features for scaling
numerical_feats = ['Age', 'BMI', 'AlcoholConsumption', 'PhysicalActivity',
                   'DietQuality', 'SleepQuality', 'SystolicBP', 'DiastolicBP',
                   'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL',
                   'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment',
                   'ADL']

if __name__ == '__main__':
    print('Importing Data')
    raw_df = csv_to_df("alzheimers_disease_data.csv")

    print('Cleaning data')
    cleaned_df = clean(raw_df, cols_to_drop)
    print('Splitting Data')
    X_train, X_test, y_train, y_test = split_data(cleaned_df, 'Diagnosis', test_size= 0.2)
    print('Training Classifiers')
    naive_bayes_clf, log_reg_clf, dt_clf, rf_clf = train_pipe(X_train, y_train, numerical_feats, categorical_feats)
    print('Evaluating Model Performance')
    nb_metrics = evaluate(naive_bayes_clf, X_test, y_test)
    lr_metrics = evaluate(log_reg_clf, X_test, y_test)
    dt_metrics = evaluate(dt_clf, X_test, y_test)
    rf_metrics = evaluate(rf_clf, X_test, y_test)
    print('Evaluations Metrics for Naive Bayes clf:')
    print(nb_metrics)
    print('Evaluations Metrics for Logistic Regression clf:')
    print(lr_metrics)
    print('Evaluations Metrics for Decision Tree clf:')
    print(dt_metrics)
    print('Evaluations Metrics for Random Forest clf:')
    print(rf_metrics)
