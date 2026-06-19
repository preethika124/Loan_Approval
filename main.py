import pandas as pd
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)



df = pd.read_csv("Finance.csv")


# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.isna().sum()*100/len(df))

# sns.countplot(x = df['Gender'])
# plt.show()

#replacing null values of gender with most frequent occuring

df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])


# sns.countplot(x= df['Married'])
# plt.show()
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])


# sns.countplot(x=df['Dependents'])

df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])

# sns.countplot(x = df['Self_Employed'])

df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

# print(df.isna().sum()*100/len(df))
# plt.show()


df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())


# sns.countplot(x = df['Credit_History'])
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
# plt.show()

# print(df.isna().sum())

df.replace({
    "Loan_Status": {'N': 0, 'Y': 1},
    "Gender": {'Male': 0, 'Female': 1},
    "Education": {'Not Graduate': 0, 'Graduate': 1},
    "Married": {'No': 0, 'Yes': 1},
    "Self_Employed": {'No': 0, 'Yes': 1}
}, inplace=True)

df["Loan_Status"] = df["Loan_Status"].astype(int)
df["Gender"] = df["Gender"].astype(int)
df["Education"] = df["Education"].astype(int)
df["Married"] = df["Married"].astype(int)
df["Self_Employed"] = df["Self_Employed"].astype(int)
df = pd.get_dummies(
    df,
    columns=["Property_Area", "Dependents"],
    dtype=int
)

# print(df.head())


def train_test_split_and_features(df):
    y = df["Loan_Status"]
    x = df.drop(['Loan_Status', 'Loan_ID'], axis=1)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state = 0)
    # print(x.head(5))
    # print(x.columns)
    features = list(x.columns)
    return x_train, x_test, y_train, y_test,features


x_train, x_test, y_train, y_test,features = train_test_split_and_features(df)

def fit_and_evaluate_model(x_train, x_test, y_train, y_test):
    random_forest =  RandomForestClassifier(random_state=0,class_weight="balanced",\
                                            max_depth=5,\
                                            min_samples_split= 0.01,\
                                            max_features= 0.8,
                                            max_samples= 0.8)

    model = random_forest.fit(x_train, y_train)
    random_forest_predict = random_forest.predict(x_test)
    random_forest_conf_matrix = confusion_matrix(y_test, random_forest_predict)
    random_forest_acc_score = accuracy_score(y_test, random_forest_predict)
    print("confussion matrix")
    print(random_forest_conf_matrix)
    print("\n")
    print("Accuracy of Random Forest:",random_forest_acc_score*100,'\n')
    print(classification_report(y_test,random_forest_predict))
    return model

# model = fit_and_evaluate_model(x_train, x_test, y_train, y_test)
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    roc_auc_score
)

def evaluate_model(model, x_train, x_test, y_train, y_test):

    model.fit(x_train, y_train)

    pred = model.predict(x_test)

    probs = model.predict_proba(x_test)[:,1]

    print("Confusion Matrix")
    print(confusion_matrix(y_test, pred))

    print("\nAccuracy")
    print(accuracy_score(y_test, pred))

    print("\nROC AUC")
    print(roc_auc_score(y_test, probs))

    print("\nClassification Report")
    print(classification_report(y_test, pred))

    return model

from sklearn.linear_model import LogisticRegression

# model = LogisticRegression(
#     max_iter=1000,
#     class_weight='balanced',
#     random_state=0
# )

# evaluate_model(
#     model,
#     x_train,
#     x_test,
#     y_train,
#     y_test
# )

# from sklearn.tree import DecisionTreeClassifier

# model = DecisionTreeClassifier(
#     max_depth=5,
#     class_weight='balanced',
#     random_state=0
# )

# evaluate_model(
#     model,
#     x_train,
#     x_test,
#     y_train,
#     y_test
# )






from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=0,
    eval_metric='logloss'
)

evaluate_model(
    model,
    x_train,
    x_test,
    y_train,
    y_test
)



# from sklearn.svm import SVC

# model = SVC(
#     probability=True,
#     class_weight='balanced',
#     random_state=0
# )

# evaluate_model(
#     model,
#     x_train,
#     x_test,
#     y_train,
#     y_test
# )


# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()

# x_train_scaled = scaler.fit_transform(x_train)
# x_test_scaled = scaler.transform(x_test)
# from sklearn.neural_network import MLPClassifier
# model = MLPClassifier(
#     hidden_layer_sizes=(64, 32),
#     activation='relu',
#     solver='adam',
#     max_iter=1000,
#     random_state=0
# )

# evaluate_model(
#     model,
#     x_train_scaled,
#     x_test_scaled,
#     y_train,
#     y_test
# )


probs = model.predict_proba(x_test)[:,1]


from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, probs)

j_scores = tpr - fpr

best_idx = np.argmax(j_scores)

best_threshold = thresholds[best_idx]

print("Best ROC Threshold:", best_threshold)







rf_proba = model.predict_proba(x_test)
pred = (probs >= best_threshold).astype(int)
print(rf_proba[0:10])
print(pred)


importances = pd.DataFrame(model.feature_importances_)
importances['features'] = features
importances.columns = ['importance','feature']
importances.sort_values(by = 'importance', ascending= True,inplace=True)

import matplotlib.pyplot as plt
plt.barh(importances.feature, importances.importance)
# plt.show()


import joblib

joblib.dump(model, "loan_model.pkl")
joblib.dump(features, "features.pkl")
joblib.dump(best_threshold, "threshold.pkl")