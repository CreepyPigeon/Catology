import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class CatBreedAnalyzer:
    def __init__(self, file_path):
        # self.data is a dataframe
        self.data = pd.read_excel(file_path)
        self._remove_columns()
        self.analyzed_data = None

    def _remove_columns(self):
        return self.data.iloc[:, 2:-1]

    def analyze_missing_values(self):
        missing_values = self.data.isnull().sum()
        print("Missing Values:\n", missing_values)

    def check_repeated_instances(self):
        repeated_instances = self.data[self.data.duplicated()]
        print("Repeated Instances:\n", repeated_instances)

    def count_instances_per_breed(self):
        breed_counts = self.data['Race'].value_counts()
        print("Number of Instances for Each Breed:\n", breed_counts)

    def extract_distinct_values(self):
        for column in self.data.columns:
            distinct_values = self.data[column].value_counts()
            print(f"Distinct Values for {column}:\n", distinct_values)
            print(f"Total distinct values for {column}: {distinct_values.count()}")

    def check_balance(self):
        breed_counts = self.data['Race'].value_counts()
        balanced = breed_counts.min() == breed_counts.max()
        print(f"Dataset is balanced: {balanced}")

    def visualize_distribution(self):
        for column in self.data.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.data[column], bins=30, kde=True)
            plt.title(f"Histogram of {column}")
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.show()

    def build_correlation_matrix(self):

        correlation_matrix = self.data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix")
        plt.show()


if __name__ == "__main__":
    analyzer = CatBreedAnalyzer("data\processed\Translated_Cat_Dataset.xlsx")
    analyzer.analyze_missing_values()
    analyzer.check_repeated_instances()
    analyzer.count_instances_per_breed()
    analyzer.extract_distinct_values()
    analyzer.check_balance()
    analyzer.visualize_distribution()
    analyzer.build_correlation_matrix()