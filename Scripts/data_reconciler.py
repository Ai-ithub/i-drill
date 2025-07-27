import pandas as pd
import numpy as np
import logging
from sklearn.impute import SimpleImputer

# تنظیم لاگ‌گیری
logging.basicConfig(
    filename='data_reconciliation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DataReconciler:
    """کلاسی برای تصحیح داده‌های گمشده و صاف کردن داده‌های پرنویز."""
    
    def __init__(self, id_column='id', timestamp_column='timestamp'):
        self.id_column = id_column
        self.timestamp_column = timestamp_column
        self.logger = logging.getLogger(__name__)

    def impute_missing_values(self, df):
        """پر کردن داده‌های گمشده با استفاده از Linear Interpolation."""
        df = df.copy()
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in [self.id_column, self.timestamp_column]:
                missing_before = df[col].isna().sum()
                if missing_before > 0:
                    # استفاده از Linear Interpolation
                    df[col] = df[col].interpolate(method='linear', limit_direction='both')
                    missing_after = df[col].isna().sum()
                    for idx, val in df[col].items():
                        if pd.isna(df.loc[idx, col]) and not pd.isna(val):
                            self.logger.info(
                                f"Value at timestamp {df.loc[idx, self.timestamp_column]} "
                                f"in column {col} imputed from NaN to {val:.4f}"
                            )
                    if missing_after > 0:
                        # اگر هنوز NaN باقی مونده، با میانگین پر کن
                        imputer = SimpleImputer(strategy='mean')
                        df[col] = imputer.fit_transform(df[[col]])
                        for idx, val in df[col].items():
                            if pd.isna(df.loc[idx, col]) and not pd.isna(val):
                                self.logger.info(
                                    f"Value at timestamp {df.loc[idx, self.timestamp_column]} "
                                    f"in column {col} imputed from NaN to {val:.4f} (mean)"
                                )
        return df

    def smooth_data(self, df, span=3):
        """صاف کردن داده‌ها با Exponential Moving Average."""
        df = df.copy()
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in [self.id_column, self.timestamp_column]:
                original_values = df[col].copy()
                df[col] = df[col].ewm(span=span, adjust=False).mean()
                for idx, (orig, smoothed) in enumerate(zip(original_values, df[col])):
                    if not pd.isna(orig) and abs(orig - smoothed) > 1e-6:
                        self.logger.info(
                            f"Value at timestamp {df.loc[idx, self.timestamp_column]} "
                            f"in column {col} smoothed from {orig:.4f} to {smoothed:.4f}"
                        )
        return df

    def reconcile(self, df):
        """اجرای فرآیند تصحیح داده‌ها (پر کردن گمشده‌ها و صاف کردن)."""
        if self.id_column not in df.columns:
            raise ValueError(f"Column {self.id_column} not found in DataFrame")
        if self.timestamp_column not in df.columns:
            raise ValueError(f"Column {self.timestamp_column} not found in DataFrame")

        # پر کردن داده‌های گمشده
        df = self.impute_missing_values(df)
        # صاف کردن داده‌ها
        df = self.smooth_data(df)
        return df
