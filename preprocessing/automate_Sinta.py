from sklearn.base import BaseEstimator, TransformerMixin

class ValueReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, replacements: dict):
        self.replacements = replacements

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, replace_map in self.replacements.items():
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].replace(replace_map)
        return X_copy


class SklearnPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, num_columns, ordinal_columns, nominal_columns, degree=2):
        self.num_columns = num_columns
        self.ordinal_columns = ordinal_columns
        self.nominal_columns = nominal_columns
        self.degree = degree

        # Custom replacement
        self.replacements = {
            'Item_Fat_Content': {
                'LF': 'Low Fat',
                'low fat': 'Low Fat',
                'reg': 'Regular'
            }
        }

    def fit(self, X, y=None):
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OrdinalEncoder, OneHotEncoder
        from sklearn.compose import ColumnTransformer

        self.replacer = ValueReplacer(self.replacements)

        self.num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('poly', PolynomialFeatures(degree=self.degree, include_bias=False)),
            ('scaler', StandardScaler())
        ])

        self.ordinal_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ord_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        self.nominal_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('nom_encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer([
            ('num_pipeline', self.num_pipeline, self.num_columns),
            ('ordinal_pipeline', self.ordinal_pipeline, self.ordinal_columns),
            ('nominal_pipeline', self.nominal_pipeline, self.nominal_columns)
        ]).set_output(transform='pandas')

        X_clean = self.replacer.transform(X)
        self.preprocessor.fit(X_clean)
        return self

    def transform(self, X):
        X_clean = self.replacer.transform(X)
        return self.preprocessor.transform(X_clean)
