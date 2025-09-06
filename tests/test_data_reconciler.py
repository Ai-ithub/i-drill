import pandas as pd
import numpy as np
from src.i_drill.processing.data_reconciler import DataReconciler

import unittest

class TestDataReconciler(unittest.TestCase):
    def setUp(self):
        # دیتاست نمونه برای تست
        self.df = pd.DataFrame({
            'id': [1001, 1002, 1003, 1004, 1005],
            'timestamp': pd.to_datetime(['2023-01-01 00:00', '2023-01-01 00:01',
                                        '2023-01-01 00:02', '2023-01-01 00:03',
                                        '2023-01-01 00:04']),
            'sensor1': [10.0, np.nan, 12.0, 15.0, np.nan],
            'sensor2': [20.0, 21.0, 22.0, np.nan, 24.0]
        })
        self.reconciler = DataReconciler(id_column='id', timestamp_column='timestamp')

    def test_impute_missing_values(self):
        df_imputed = self.reconciler.impute_missing_values(self.df)
        self.assertFalse(df_imputed['sensor1'].isna().any(), "sensor1 should have no NaN after imputation")
        self.assertFalse(df_imputed['sensor2'].isna().any(), "sensor2 should have no NaN after imputation")
        self.assertAlmostEqual(df_imputed.loc[1, 'sensor1'], 11.0, places=2,
                              msg="sensor1 at index 1 should be interpolated to 11.0")

    def test_smooth_data(self):
        df_imputed = self.reconciler.impute_missing_values(self.df)
        df_smoothed = self.reconciler.smooth_data(df_imputed, span=3)
        self.assertFalse(df_smoothed['sensor1'].isna().any(), "sensor1 should have no NaN after smoothing")
        # بررسی صاف شدن داده‌ها (مقدار تقریبی)
        expected_sensor1 = df_imputed['sensor1'].ewm(span=3, adjust=False).mean()
        for idx in range(len(df_smoothed)):
            self.assertAlmostEqual(df_smoothed.loc[idx, 'sensor1'], expected_sensor1[idx], places=4,
                                  msg=f"sensor1 at index {idx} not smoothed correctly")

    def test_reconcile(self):
        df_reconciled = self.reconciler.reconcile(self.df)
        self.assertFalse(df_reconciled.isna().any().any(), "No NaN values should remain after reconciliation")
        self.assertTrue((df_reconciled['id'] == self.df['id']).all(), "ID column should remain unchanged")
        self.assertTrue((df_reconciled['timestamp'] == self.df['timestamp']).all(),
                        "Timestamp column should remain unchanged")

if __name__ == '__main__':
    unittest.main()
