from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.25)

# predictive model

df["Q16"] = df["Q16"].fillna(0)
cols = ["Q16"]
df[cols] = df[cols].applymap(np.int64)
print(df)
df["Q4"] = df["Q4"].fillna(0)
cols = ["Q4"]
df[cols] = df[cols].applymap(np.int64)
print(df)
df["Q10"] = df["Q10"].fillna(0)
cols = ["Q10"]
df[cols] = df[cols].applymap(np.int64)
print(df)
df["Q13"] = df["Q13"].fillna(0)
cols = ["Q13"]
df[cols] = df[cols].applymap(np.int64)
print(df)
df["Q14"] = df["Q14"].fillna(0)
cols = ["Q14"]
df[cols] = df[cols].applymap(np.int64)
print(df)


from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.40)

train_y = train["Q16"]
test_y = test["Q16"]

train_x = train
train_x.pop("Q16")
test_x = test
test_x.pop("Q16")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

logisticRegr = LogisticRegression()
logisticRegr.fit(X=train_x, y=train_y)

test_y_pred = logisticRegr.predict(test_x)
confusion_matrix = confusion_matrix(test_y, test_y_pred)
print("Intercept: " + str(logisticRegr.intercept_))
print("Regression: " + str(logisticRegr.coef_))
print(
    "Accuracy of logistic regression classifier on test set: {:.2f}".format(
        logisticRegr.score(test_x, test_y)
    )
)
print(classification_report(test_y, test_y_pred))

confusion_matrix_df = pd.DataFrame(
    confusion_matrix, ("No churn", "Churn"), ("No churn", "Churn")
)
heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={"size": 20}, fmt="d")
heatmap.yaxis.set_ticklabels(
    heatmap.yaxis.get_ticklabels(), rotation=0, ha="right", fontsize=14
)
heatmap.xaxis.set_ticklabels(
    heatmap.xaxis.get_ticklabels(), rotation=45, ha="right", fontsize=14
)
plt.ylabel("True label", fontsize=14)
plt.xlabel("Predicted label", fontsize=14)

df["Q16"].value_counts()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

X = df.drop("Q16", axis=1)
y = df["Q16"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=20
)

knn = KNeighborsClassifier()

knn_param_grid = {"n_neighbors": np.arange(5, 26), "weights": ["uniform", "distance"]}

knn_cv = GridSearchCV(knn, param_grid=knn_param_grid, cv=5)

## Fit knn to training data
knn_cv.fit(X_train, y_train)

## Get info about best hyperparameters
print("Tuned KNN Parameters: {}".format(knn_cv.best_params_))
print("Best KNN Training Score:{}".format(knn_cv.best_score_))

## Predict knn on test data
print("KNN Test Performance: {}".format(knn_cv.score(X_test, y_test)))

## Obtain model performance metrics
knn_pred_prob = knn_cv.predict_proba(X_test)[:, 1]
knn_auroc = roc_auc_score(y_test, knn_pred_prob)
print("KNN AUROC: {}".format(knn_auroc))
knn_y_pred = knn_cv.predict(X_test)
print(classification_report(y_test, knn_y_pred))

## Instantiate classifier
lr = LogisticRegression(random_state=30)

## Set up hyperparameter grid for tuning
lr_param_grid = {"C": [0.0001, 0.001, 0.01, 0.05, 0.1]}

## Tune hyperparamters
lr_cv = GridSearchCV(lr, param_grid=lr_param_grid, cv=5)

## Fit lr to training data
lr_cv.fit(X_train, y_train)

## Get info about best hyperparameters
print("Tuned LR Parameters: {}".format(lr_cv.best_params_))
print("Best LR Training Score:{}".format(lr_cv.best_score_))

## Predict lr on test data
print("LR Test Performance: {}".format(lr_cv.score(X_test, y_test)))

## Obtain model performance metrics
lr_pred_prob = lr_cv.predict_proba(X_test)[:, 1]
lr_auroc = roc_auc_score(y_test, lr_pred_prob)
print("LR AUROC: {}".format(lr_auroc))
lr_y_pred = lr_cv.predict(X_test)

## Instatiate classifier
rf = RandomForestClassifier(random_state=30)

## Set up hyperparameter grid for tuning
rf_param_grid = {
    "n_estimators": [200, 250, 300, 350, 400, 450, 500],
    "max_features": ["sqrt", "log2"],
    "max_depth": [3, 4, 5, 6, 7],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4],
}

## Tune hyperparameters
rf_cv = RandomizedSearchCV(
    rf, param_distributions=rf_param_grid, cv=5, random_state=30, n_iter=20
)

## Fit RF to training data
rf_cv.fit(X_train, y_train)

## Get info about best hyperparameters
print("Tuned RF Parameters: {}".format(rf_cv.best_params_))
print("Best RF Training Score:{}".format(rf_cv.best_score_))

## Predict RF on test data
print("RF Test Performance: {}".format(rf_cv.score(X_test, y_test)))

## Obtain model performance metrics
rf_pred_prob = rf_cv.predict_proba(X_test)[:, 1]
rf_auroc = roc_auc_score(y_test, rf_pred_prob)
print("RF AUROC: {}".format(rf_auroc))
rf_y_pred = rf_cv.predict(X_test)
print(classification_report(y_test, rf_y_pred))

sgb = GradientBoostingClassifier(random_state=30)

## Set up hyperparameter grid for tuning
sgb_param_grid = {
    "n_estimators": [200, 300, 400, 500],
    "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
    "max_depth": [3, 4, 5, 6, 7],
    "min_samples_split": [2, 5, 10, 20],
    "min_weight_fraction_leaf": [0.001, 0.01, 0.05],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1],
    "max_features": ["sqrt", "log2"],
}

## Tune hyperparamters
sgb_cv = RandomizedSearchCV(
    sgb, param_distributions=sgb_param_grid, cv=5, random_state=30, n_iter=20
)

## Fit SGB to training data
sgb_cv.fit(X_train, y_train)

## Get info about best hyperparameters
print("Tuned SGB Parameters: {}".format(sgb_cv.best_params_))
print("Best SGB Training Score:{}".format(sgb_cv.best_score_))

## Predict SGB on test data
print("SGB Test Performance: {}".format(sgb_cv.score(X_test, y_test)))

## Obtain model performance metrics
sgb_pred_prob = sgb_cv.predict_proba(X_test)[:, 1]
sgb_auroc = roc_auc_score(y_test, sgb_pred_prob)
print("SGB AUROC: {}".format(sgb_auroc))
sgb_y_pred = sgb_cv.predict(X_test)
print(classification_report(y_test, sgb_y_pred))


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
y_model = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("SGB accuracy score is:", accuracy_score(y_test, y_model))

from sklearn.metrics import confusion_matrix

mat = confusion_matrix(y_test, y_model)

sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value');


import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='linear', C=0.01).fit(X_train, y_train)

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
titles_options = [("SVC Confusion matrix, without normalization", None),
                  ("SVC Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                            
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()
