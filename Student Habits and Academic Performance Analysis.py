import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os

class DataLoadError(Exception):
    pass

class DataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load_data(self):
        """Loads dataset from the local file path"""
        if not os.path.exists(self.dataset_path):
            raise DataLoadError(f"File not found: {self.dataset_path}")
        try:
            return pd.read_csv(self.dataset_path)
        except Exception as e:
            raise DataLoadError(f"Failed to load data: {str(e)}")

class DataCleaner:
    def __init__(self, df):
        """Initialize dataframe"""
        self.df = df

    def validate_columns(self, required_columns):
        """Check if required columns exist in the DataFrame"""
        missing_cols = [col for col in required_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def clean(self):
        """Performs data validation and cleaning"""
        required_columns = ['mental_health_rating', 'exam_score', 'study_hours_per_day', 'sleep_hours']
        self.validate_columns(required_columns)
        
        # Handle missing values based on column type
        for column in self.df.columns:
            if self.df[column].dtype in ['float64', 'int64']:
                self.df[column].fillna(self.df[column].mean(), inplace=True)
            else:
                self.df[column].fillna(self.df[column].mode()[0], inplace=True)
        return self.df

class StudentAnalyzer:
    @staticmethod
    def score_by_health(df):
        """Calculate mean final_score by mental_health group"""
        return df.groupby('mental_health_rating')['exam_score'].mean()

    @staticmethod
    def detect_outliers(df, column):
        """Detect outliers using IQR method"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        return df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]

class VisualizationEngine:
    def __init__(self, df):
        self.df = df

    def plot_histogram(self, column, save_path=None):
        """Plot histogram for a specified column"""
        plt.figure()
        plt.hist(self.df[column], bins=20)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        if save_path:
            plt.savefig(save_path)
        plt.show()

class CustomVisualizer(VisualizationEngine):
    def plot_correlation_heatmap(self, save_path=None):
        """Plot correlation heatmap for numerical columns"""
        plt.figure()
        numerical_df = self.df.select_dtypes(include=['float64', 'int64'])
        sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        if save_path:
            plt.savefig(save_path)
        plt.show()

class ScorePredictor:
    def __init__(self, df):
        self.model = LinearRegression()
        X = df[['study_hours_per_day', 'sleep_hours']]
        y = df['exam_score']
        self.model.fit(X, y)

    def predict(self, study_hours_per_day, sleep_hours):
        """Predict final_score based on study_time and sleep"""
        if study_hours_per_day < 0 or sleep_hours < 0:
            raise ValueError("Inputs must be positive")
        if study_hours_per_day > 168 or sleep_hours > 24:
            raise ValueError("Study time must be <= 168 hours and sleep <= 24 hours")
        return self.model.predict([[study_hours_per_day, sleep_hours]])[0]

# Dataset
if __name__ == "__main__":
    file_path = r"C:\Users\hp\Downloads\archive (1)\student_habits_performance.csv"
    
    # Load and clean data
    loader = DataLoader(file_path)
    df = loader.load_data()
    cleaner = DataCleaner(df)
    df_clean = cleaner.clean()

    # Analyze data
    analyzer = StudentAnalyzer()
    mean_scores = analyzer.score_by_health(df_clean)
    outliers_study_time = analyzer.detect_outliers(df_clean, 'study_hours_per_day')

    # Visualize data
    vis = CustomVisualizer(df_clean)
    vis.plot_histogram('exam_score', save_path='histogram_final_score.png')
    vis.plot_correlation_heatmap(save_path='correlation_heatmap.png')

     # Predict score
    predictor = ScorePredictor(df_clean)
    try:
        predicted_score = predictor.predict(study_hours_per_day=10, sleep_hours=7)  # Fixed parameter names
        print(f"Predicted score for 10 hours study and 7 hours sleep: {predicted_score:.2f}")
    except ValueError as e:
        print(f"Prediction error: {e}")

