from loaders import csv_to_df
from clean import clean, split_data
from train import train_pipe
from eval import evaluate

cols_to_drop = ['PatientID', 'DoctorInCharge']
categorical_feats = ['Gender', 'Ethnicity', 'EducationLevel', 'Smoking',
                     'FamilyHistoryAlzheimers', 'CardiovascularDisease',
                     'Diabetes', 'Depression', 'HeadInjury', 'Hypertension',
                     'MemoryComplaints', 'BehavioralProblems', 'Confusion',
                     'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks',
                     'Forgetfulness']

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
    naive_bayes_clf, log_reg_clf = train_pipe(X_train, y_train, numerical_feats, categorical_feats)
    print('Evaluating Model Performance')
    nb_metrics = evaluate(naive_bayes_clf, X_test, y_test)
    lr_metrics = evaluate(log_reg_clf, X_test, y_test)
    print('Evaluations Metrics for Naive Bayes clf:')
    print(nb_metrics)
    print('Evaluations Metrics for Logistic Regression clf:')
    print(lr_metrics)
