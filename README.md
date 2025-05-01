The goal of this project is to train and compare the performance of multiple machine learning
classifiers, that predicted if a patient was diagnosed with Alzheimer's disease, based on their health information.

To accomplish this, I wrote functions that do the following:

Converting csv -> pandas dataframe

Clean DF by dropping non-relevant features (patient_ID and DoctorInCharge)

Splitting df into training and testing set

Created pipeline that scaled the numerical features, and encoded categorical features

After feature scaling/encoding, used grid_search to find optimal parameters, and returned best fitted model

Function that evaluates a models performance, and returns evaluations metrics: Precision, recall and f1_score
