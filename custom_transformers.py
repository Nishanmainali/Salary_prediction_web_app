from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer

class MultiLabelBinarizerDF(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, delimiter=';', top_n=10):
        self.columns = columns
        self.delimiter = delimiter
        self.top_n = top_n
        self.top_labels_ = {}            # Stores top-N labels per column
        self.label_binarizers = {}       # Stores the MultiLabelBinarizer per column

    def fit(self, X, y=None):
        for col in self.columns:
            # Flatten all items in the column
            all_items = X[col].dropna().apply(
                lambda x: [i.strip() for i in str(x).split(self.delimiter)]
            ).explode()

            # Get top-N labels
            top_items = [item for item, _ in Counter(all_items).most_common(self.top_n)]
            self.top_labels_[col] = top_items

            # Prepare cleaned labels for fitting
            cleaned = X[col].dropna().apply(lambda x: self._clean_labels(x, top_items))
            mlb = MultiLabelBinarizer()
            mlb.fit(cleaned)
            self.label_binarizers[col] = mlb

        return self
    def transform(self, X):
        print("✅ New MultiLabelBinarizerDF.transform() called")
        encoded_frames = []
        for col in self.columns:
            cleaned = X[col].fillna('').apply(
                lambda x: self._clean_labels(x, self.top_labels_[col])
            )
            encoded = self.label_binarizers[col].transform(cleaned)
            columns = [f"{col}_{cls}" for cls in self.label_binarizers[col].classes_]
            df_encoded = pd.DataFrame(encoded, columns=columns, index=X.index)
            encoded_frames.append(df_encoded)
        return pd.concat(encoded_frames, axis=1)

    def _clean_labels(self, val, top_items):
        # Split and strip
        items = [i.strip() for i in str(val).split(self.delimiter) if i.strip()]

        # Map infrequent labels to 'Other', keep unique
        cleaned = [i if i in top_items else 'Other' for i in items]
        return list(dict.fromkeys(cleaned))