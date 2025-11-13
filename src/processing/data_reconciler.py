"""
Data Reconciliation Module
Handles missing value imputation and data smoothing for sensor data.
"""
import pandas as pd
import numpy as np
import logging
from sklearn.impute import SimpleImputer
from typing import Optional

# تنظیم لاگ‌گیری
logging.basicConfig(
    filename='data_reconciliation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class DataReconciler:
    """
    Class for correcting missing data and smoothing noisy data.
    
    Provides methods for imputing missing values using linear interpolation
    and mean imputation, and smoothing data using exponential moving averages.
    
    Attributes:
        id_column: Name of the ID column in the DataFrame
        timestamp_column: Name of the timestamp column in the DataFrame
        logger: Logger instance for logging operations
    """
    
    def __init__(self, id_column: str = 'id', timestamp_column: str = 'timestamp'):
        """
        Initialize DataReconciler.
        
        Args:
            id_column: Name of the ID column in DataFrames (default: 'id')
            timestamp_column: Name of the timestamp column in DataFrames (default: 'timestamp')
        """
        self.id_column = id_column
        self.timestamp_column = timestamp_column
        self.logger = logging.getLogger(__name__)

    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values using Linear Interpolation.
        
        First attempts to fill missing values using linear interpolation.
        If any values remain missing after interpolation, fills them with
        the column mean using SimpleImputer.
        
        Args:
            df: DataFrame containing sensor data with potential missing values
            
        Returns:
            DataFrame with missing values imputed
        """
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

    def smooth_data(self, df: pd.DataFrame, span: int = 3) -> pd.DataFrame:
        """
        Smooth data using Exponential Moving Average (EMA).
        
        Applies exponential moving average smoothing to all numeric columns
        to reduce noise in sensor readings.
        
        Args:
            df: DataFrame containing sensor data to smooth
            span: Span parameter for EMA (default: 3). Larger values result in more smoothing.
            
        Returns:
            DataFrame with smoothed values
        """
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

    def reconcile(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the complete data reconciliation process.
        
        Performs both missing value imputation and data smoothing in sequence.
        Validates that required columns (id_column and timestamp_column) exist.
        
        Args:
            df: DataFrame containing sensor data to reconcile
            
        Returns:
            DataFrame with reconciled data (missing values imputed and data smoothed)
            
        Raises:
            ValueError: If required columns (id_column or timestamp_column) are missing
        """
        if self.id_column not in df.columns:
            raise ValueError(f"Column {self.id_column} not found in DataFrame")
        if self.timestamp_column not in df.columns:
            raise ValueError(f"Column {self.timestamp_column} not found in DataFrame")

        # پر کردن داده‌های گمشده
        df = self.impute_missing_values(df)
        # صاف کردن داده‌ها
        df = self.smooth_data(df)
        return df
