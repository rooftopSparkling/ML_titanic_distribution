from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class TitanicPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        df = X.copy()

        # Title
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        self.title_mapping_ = {
            "Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3,
            "Dr": 4, "Rev": 4, "Col": 4, "Major": 4, "Mlle": 4, "Countess": 4,
            "Ms": 4, "Lady": 4, "Jonkheer": 4, "Don": 4, "Dona": 4, "Mme": 4,
            "Capt": 4, "Sir": 4
        }
        df['Title'] = df['Title'].map(self.title_mapping_)

        # Title별 Age 중앙값
        self.age_medians_ = df.groupby("Title")["Age"].median()

        # Pclass별 Fare 중앙값
        self.fare_medians_ = df.groupby("Pclass")["Fare"].median()

        # Pclass별 Cabin 문자형 매핑 후 중앙값
        df['Cabin'] = df['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else np.nan)
        cabin_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8}
        df['Cabin'] = df['Cabin'].map(cabin_map)
        self.cabin_medians_ = df.groupby("Pclass")["Cabin"].median()

        return self

    def transform(self, X):
        df = X.copy()

        # Title
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].map(self.title_mapping_)
        df.drop('Name', axis=1, inplace=True)

        df['Sex'] = df['Sex'].map({"male": 0, "female": 1})

        df["Age"] = df["Age"].copy()
        for title, median in self.age_medians_.items():
            df.loc[(df['Title'] == title) & (df['Age'].isnull()), 'Age'] = median

        df.loc[df['Age'] <= 18, 'Age'] = 0
        df.loc[(df['Age'] > 18) & (df['Age'] <= 27), 'Age'] = 1
        df.loc[(df['Age'] > 27) & (df['Age'] <= 35), 'Age'] = 2
        df.loc[(df['Age'] > 35) & (df['Age'] <= 57), 'Age'] = 3
        df.loc[df['Age'] > 57, 'Age'] = 4

        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

        df.drop('Ticket', axis=1, inplace=True)

        df["Fare"] = df["Fare"].copy()
        for pclass, median in self.fare_medians_.items():
            df.loc[(df['Pclass'] == pclass) & (df['Fare'].isnull()), 'Fare'] = median

        df.loc[df['Fare'] <= 29, 'Fare'] = 0
        df.loc[(df['Fare'] > 29) & (df['Fare'] <= 70), 'Fare'] = 1
        df.loc[(df['Fare'] > 70) & (df['Fare'] <= 110), 'Fare'] = 2
        df.loc[df['Fare'] > 110, 'Fare'] = 3

        df['Cabin'] = df['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else np.nan)
        cabin_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8}
        df['Cabin'] = df['Cabin'].map(cabin_map)

        for pclass, median in self.cabin_medians_.items():
            df.loc[(df['Pclass'] == pclass) & (df['Cabin'].isnull()), 'Cabin'] = median

        df.drop('Embarked', axis=1, inplace=True)

        return df

