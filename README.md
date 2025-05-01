The goal of this project is to train and compare the performance of multiple machine learning
classifiers, that predicted if a patient was diagnosed with Alzheimer's disease, based on their health information.

To accomplish this, I wrote functions that do the following:< br / >
Converting csv -> pandas dataframe< br / >
Clean DF by dropping non-relevant features (patient_ID and DoctorInCharge)< br / >
Splitting df into training and testing set< br / >
Created pipeline that scaled the numerical features, and encoded categorical features< br / >
After feature scaling/encoding, used grid_search to find optimal parameters, and returned best fitted model< br / >
Function that evaluates a models performance, and returns evaluations metrics: Precision, recall and f1_score< br / >
