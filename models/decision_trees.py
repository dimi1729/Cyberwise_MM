from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRFClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from analysis.preprocessing import split_cic_data
from analysis.plotting import plot_confusion_matrix, plot_feature_importance

from typing import Dict, Optional
class RandomForestAnalyzer:
    def __init__(self, data_path: str):
        self.df = pd.read_csv(data_path)
        self.feature_names = self.df.drop(columns=['Label']).columns.tolist()
        self._prepare_data()
        
    def _prepare_data(self):
        splits = split_cic_data(self.df, 0.7, 0.15, 0.15)
        self.X_train = splits[0].numpy()
        self.X_test = splits[2].numpy() 
        self.y_train = splits[3].numpy().ravel()
        self.y_test = splits[5].numpy().ravel()
        
    def train(self, max_depth: int, class_weight: Dict[int, float]):
        # Class weights should be in the form {0: negative_weight, 1: positive_weight}
        # Ex. {0: 1, 1:10}
        self.model = RandomForestClassifier(
            n_estimators=100,
            class_weight=class_weight,
            max_depth=max_depth,
            random_state=1729
        )
        self.model.fit(self.X_train, self.y_train)
        
    def evaluate(self, name: Optional[str] = None):
        preds = self.model.predict(self.X_test)
        print("Random Forest Results:")
        print(classification_report(self.y_test, preds))
        plot_confusion_matrix(
            confusion_matrix(self.y_test, preds),
            class_names=["Benign", "Malicious"],
            name=name
        )
        
    def plot_importance(self, name: Optional[str] = None):
        """Feature importance visualization"""
        plot_feature_importance(
            self.model.feature_importances_,
            self.feature_names,
            name=name
        )

class XGBoostRFAnalyzer:
    def __init__(self, data_path: str):
        self.df = pd.read_csv(data_path)
        self.feature_names = self.df.drop(columns=['Label']).columns.tolist()
        self._prepare_data()
        
    def _prepare_data(self):
        """Identical data prep to RF class"""
        splits = split_cic_data(self.df, 0.7, 0.15, 0.15)
        self.X_train = splits[0].numpy()
        self.X_test = splits[2].numpy()
        self.y_train = splits[3].numpy().ravel()
        self.y_test = splits[5].numpy().ravel()
        
    def train(self):
        """XGBoost Random Forest with custom params"""
        self.model = XGBRFClassifier(
            n_estimators=100,
            subsample=0.8,
            colsample_bynode=0.3,
            scale_pos_weight=10,
            random_state=1729,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.model.fit(self.X_train, self.y_train)
        
    def evaluate(self, name: Optional[str] = None):
        """XGB-specific evaluation"""
        preds = self.model.predict(self.X_test)
        print("\nXGBoost RF Results:")
        print(classification_report(self.y_test, preds))
        plot_confusion_matrix(
            confusion_matrix(self.y_test, preds),
            class_names=["Benign", "Malicious"],
            name=name
        )
        
    def plot_importance(self, name: Optional[str] = None):
        """XGB-specific importance plot"""
        plot_feature_importance(
            self.model.feature_importances_,
            self.feature_names,
            name=name
        )

if __name__ == '__main__':
    # Define the path to your dataset
    data_path = "data/Merged01.csv"  # Adjust this path to your actual data location
    
    rf_analyzer = RandomForestAnalyzer(data_path)
    rf_analyzer.train(max_depth=10, class_weight={0: 15, 1: 1})
    rf_analyzer.evaluate()
    rf_analyzer.plot_importance()
    print('done rf training')

    quit()
    
    xgb_analyzer = XGBoostRFAnalyzer(data_path)
    xgb_analyzer.train()
    xgb_analyzer.evaluate()
    xgb_analyzer.plot_importance()
    print('done xgboost training')
