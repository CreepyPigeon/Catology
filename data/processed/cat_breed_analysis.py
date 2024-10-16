import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def _remove_columns(file_path):
    df = pd.read_excel(file_path)
    df = df.drop(df.columns[[0, 1, -1]], axis=1)
    df.to_csv('modified_dataset.csv', index=False)
    df_modified = pd.read_csv('modified_dataset.csv')

    return df_modified

    # oversampling documentare
    # unknown - per atribut: stergi instanta sau media sau sirul vid

def map_age(age: float):

    if isinstance(age, (int, float)):
        return age

    if age == 'Less than 1':
        return 0.5
    elif age == '1-2 years':
        return 1.5
    elif age == '2-10 years':
        return 6
    elif age == 'More than 10':
        return 15

class OriginalDatasetAnalyzer:
    def __init__(self, file_path=None):
        # TO-DO functions for getting the missing values,
        # instances that repeat, write the results to 'errors.txt' in the 'results' Directory
        pass

class CatBreedAnalyzer:
    def __init__(self, file_path=None):

        # self.data is a dataframe
        if file_path is None:
            self.data = pd.read_csv('modified_dataset.csv')
        else:
            self.data = _remove_columns(file_path)
            self.analyzed_data = None

        # self._transform_age_column()
        # self._transform_sex_column()  # F = 0; M = 1
        # self._transform_abundance_column()
        # self._transform_race_column()
        # self._transform_


    def analyze_missing_values(self):
        missing_values = self.data.isnull().sum()
        print("Missing Values:\n", missing_values)

    def check_and_drop_repeated_instances(self):
        repeated_instances = self.data[self.data.duplicated()]
        print("Repeated Instances:\n", repeated_instances)
        print('Dropping them ...')
        self.data = self.data.drop_duplicates()

    # bar plot since the data is not numerical
    def count_and_show_instances_per_breed(self):
        breed_counts = self.data['Race'].value_counts()
        total_count = breed_counts.sum()

        # for calculating the percentage for each breed
        breed_percentages = (breed_counts / total_count) * 100
        print("Number of Instances and Percentages for Each Breed:\n")

        for breed, count, percentage in zip(breed_counts.index, breed_counts.values, breed_percentages.values):
            print(f"{breed}: {count} instances, {percentage:.2f}% of total")

        # map breeds to numeric values for plotting
        breed_names = breed_counts.index
        breed_indices = range(len(breed_names))

        plt.figure(figsize=(10, 6))
        plt.bar(breed_indices, breed_counts.values, color='lightblue')

        plt.xlabel("Breed")
        plt.ylabel("Number of Instances")
        plt.title("Number of Instances per Cat Breed")
        plt.xticks(breed_indices, breed_names, rotation=45)

        plt.tight_layout()
        plt.show()

        print('\n')
        self.check_balance()

    # ASTA MAI LA FINAL
    # trebuie facut automat pentru fiecare coloana
    # pe langa histograme si boxplot-uri (de invatat de citit si astea)
    #, la final, adaugi boxplot-urile si histogramele ca imagini in Direcotrul numit results
    def extract_and_plot_distinct_values(self, column):
        if column not in self.data.columns:
            print(f"Column '{column}' does not exist in the data.")
            return

        distinct_values = self.data[column].value_counts()
        print(f"Distinct Values for {column}:\n", distinct_values)
        print(f"Total distinct values for {column}: {distinct_values.count()}")

        plt.figure(figsize=(10, 5))
        plt.bar(distinct_values.index, distinct_values.values, color='tab:blue')
        plt.title(f'Distinct Values for {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # calculate the count of occurrences for each unique value in the 'Race' column of the DataFrame.
    # return whether the minimum count of any breed is equal to the maximum count.
    def check_balance(self):
        breed_counts = self.data['Race'].value_counts()
        balanced = breed_counts.min() == breed_counts.max()
        print(f"Dataset is balanced: {balanced}")

    # KDE = Kernel Density Estimate
    """
    This function creates a histogram of the specified column in the data 
    along with a Kernel Density Estimate (KDE) curve.
     
    The KDE is a method  used to estimate the probability density function 
    of a continuous random variable. It provides a smooth curve that 
    represents the distribution of the data points.

    The KDE is useful for visualizing the underlying distribution, allowing 
    see trends and patterns that may not be immediately evident 
    from the histogram alone. For example, if you have a distribution 
    that peaks in two places, the KDE can help illustrate that bimodal 
    nature more clearly than a histogram might.
    """
    def visualize_distribution(self, column):
        if column not in self.data.columns:
            print(f"Column '{column}' does not exist in the data.")
            return

        plt.figure(figsize=(10, 6))
        sns.histplot(self.data[column], bins=30, kde=True)  # Create a histogram with KDE
        plt.title(f"Histogram of {column}")
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

    def _transform_sex_column(self):
        self.data['Sex'] = self.data['Sex'].replace({'F': 0, 'M': 1})
        self.save_to_csv()  # Save after transforming sex column

    def _transform_age_column(self):
        self.data['Age'] = self.data['Age'].apply(map_age)
        self.save_to_csv()

    def _transform_abundance_column(self):
        self.data['Abundance of natural areas'] = self.data['Abundance of natural areas'].replace('Unknown', 0)
        self.data['Abundance of natural areas'] = pd.to_numeric(self.data['Abundance of natural areas'])
        self.save_to_csv()

    """
    Factorizes the 'Race' column to get numerical and unique values
    Adds the numerical codes as a new column called 'Numerical Race'
    Renames the original 'Race' column to 'Race_Description' and keeps it
    Drops the original 'Race' column
    Reorders the columns so that 'Numerical Race' and 'Race_Description' are at the end
    """

    def _transform_race_column(self):

        if 'Numerical Race' in self.data.columns and 'Race Description' in self.data.columns:
            print("Transformation has already been applied. No further changes made.")
            return

        numerical_race, unique_races = pd.factorize(self.data['Race'])

        self.data['Numerical Race'] = numerical_race
        self.data['Race_Description'] = self.data['Race']
        self.data = self.data.drop(columns=['Race'])

        other_columns = [col for col in self.data.columns if col not in ['Numerical Race', 'Race Description']]
        self.data = self.data[other_columns + ['Numerical Race', 'Race Description']]

        self.save_to_csv()

    def add_numerical_consistency(self):
        # am nevoie si pentru place_of_living de transformat in valoare numerica
        self._transform_abundance_column()
        self._transform_race_column()

    def save_to_csv(self, file='modified_dataset.csv'):
        self.data.to_csv(file_name, index=False)

    # the correlation matrix still needs work
    def build_correlation_matrix(self):

        self.add_numerical_consistency()
        correlation_matrix = self.data.iloc[:, :-1].corr()  # Exclude last column (string representation)

        # Create a heatmap to visualize the correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix")
        plt.show()

        self.save_to_csv()


if __name__ == "__main__":

    # the full relative path
    file_name = 'modified_dataset.csv'

    # get the full path based on the current working directory
    full_file_path = os.path.join(os.getcwd(), file_name)

    # instantiate the analyzer only if the file does not exist
    if not os.path.isfile(full_file_path):
        analyzer = CatBreedAnalyzer("Translated_Cat_Dataset.xlsx")
        print("Analyzer successfully instantiated")
    else:
        print(f"{full_file_path} already exists. Analyzer not instantiated.")
        analyzer = CatBreedAnalyzer()

    # analyzer.analyze_missing_values()
    # analyzer.check_and_drop_repeated_instances()

    # + boxplot-uri
    analyzer.extract_and_plot_distinct_values('Age')

    # analyzer.count_and_show_instances_per_breed()

    # analyzer.visualize_distribution('Age')

    # YOU CAN RUN EVERYTHING EXCEPT OF THIS LITTLE FUNCTION
    # AND EVERY HELPER ASSOCIATED WITH IT
    # analyzer.build_correlation_matrix()
