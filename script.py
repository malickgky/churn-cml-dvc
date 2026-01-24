import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from imblearn.over_sampling import SMOTE
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]


def main():
    # --------------------- Data Preparation ---------------------------- #
    TRAIN_PATH = os.path.join(os.getcwd(), 'data.csv')

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"{TRAIN_PATH} not found. Make sure data.csv exists in the root folder.")

    df = pd.read_csv(TRAIN_PATH)

    df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
    df.drop(index=df[df['Age'] > 80].index.tolist(), axis=0, inplace=True)

    X = df.drop(columns=['Exited'], axis=1)
    y = df['Exited']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=45, stratify=y
    )

    # --------------------- Data Processing ---------------------------- #
    num_cols = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary']
    categ_cols = ['Gender', 'Geography']
    ready_cols = list(set(X_train.columns.tolist()) - set(num_cols) - set(categ_cols))

    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_cols)),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categ_pipeline = Pipeline([
        ('selector', DataFrameSelector(categ_cols)),
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(drop='first', sparse=False))
    ])

    ready_pipeline = Pipeline([
        ('selector', DataFrameSelector(ready_cols)),
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    all_pipeline = FeatureUnion([
        ('numerical', num_pipeline),
        ('categorical', categ_pipeline),
        ('ready', ready_pipeline)
    ])

    X_train_final = all_pipeline.fit_transform(X_train)
    X_test_final = all_pipeline.transform(X_test)

    # --------------------- Imbalancing ---------------------------- #
    vals_count = 1 - (np.bincount(y_train) / len(y_train))
    vals_count = vals_count / np.sum(vals_count)

    dict_weights = {i: vals_count[i] for i in range(2)}

    over = SMOTE(sampling_strategy=0.7)
    X_train_resampled, y_train_resampled = over.fit_resample(X_train_final, y_train)

    # --------------------- Modeling ---------------------------- #
    open('metrics.txt', 'w').close()

    def train_model(X_train, y_train, plot_name='', class_weight=None):
        clf = RandomForestClassifier(
            n_estimators=3000,
            max_depth=10,
            random_state=45,
            class_weight=class_weight
        )

        clf.fit(X_train, y_train)

        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test_final)

        score_train = f1_score(y_train, y_pred_train)
        score_test = f1_score(y_test, y_pred_test)

        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred_test)
        plt.imshow(cm, interpolation='nearest')
        plt.title(plot_name)
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        plt.xticks([0, 1], ['False', 'True'])
        plt.yticks([0, 1], ['False', 'True'])

        plt.savefig(f'{plot_name}.png', bbox_inches='tight', dpi=300)
        plt.close()

        with open('metrics.txt', 'a') as f:
            f.write(f'RandomForestClassifier {plot_name}\n')
            f.write(f"F1-score of Training is: {score_train*100:.2f} %\n")
            f.write(f"F1-Score of Validation is: {score_test*100:.2f} %\n")
            f.write('----'*10 + '\n')

        return True

    train_model(X_train_final, y_train, 'without-imbalance', None)
    train_model(X_train_final, y_train, 'with-class-weights', dict_weights)
    train_model(X_train_resampled, y_train_resampled, 'with-SMOTE', None)

    confusion_matrix_paths = [
        './without-imbalance.png',
        './with-class-weights.png',
        './with-SMOTE.png'
    ]

    plt.figure(figsize=(15, 5))
    for i, path in enumerate(confusion_matrix_paths, 1):
        img = Image.open(path)
        plt.subplot(1, len(confusion_matrix_paths), i)
        plt.imshow(img)
        plt.axis('off')

    plt.suptitle('Confusion Matrix Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('conf_matrix.png', bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    main()