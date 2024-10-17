import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()
path = os.getenv('FILE_PATH')
results_dir = os.getenv('RESULTS_DIR')

def _remove_columns(file_path):
    df = pd.read_excel(file_path)
    df = df.drop(df.columns[[0, 1, -1]], axis=1)
    df.to_excel('modified_dataset.xlsx', index=False)
    df_modified = pd.read_excel('modified_dataset.xlsx')

    return df_modified


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

def map_area(designated_area):
    if isinstance(designated_area, int):
        return designated_area

    if designated_area == 'U':
        return 1
    elif designated_area == 'R':
        return 2
    elif designated_area == 'PU':
        return 3

def map_cats(number_of_cats):
    return 6 if number_of_cats == 'Over 5' else int(number_of_cats)

def map_sex(sex):

    if sex == 'F':
        return 0
    elif sex == 'M':
        return 1

    return sex


def map_place_of_living(place):
    if place == "ASB":
        return 1
    elif place == "AAB":
        return 2
    elif place == "ML":
        return 3
    elif place == "MI":
        return 4
    else:
        return place


class OriginalDatasetAnalyzer:
    def __init__(self, file_path=None):
        self.data = _remove_columns(file_path)
        self.analyzed_data = None
        self.results_directory = r"Catology\data\results"
        self.errors_file = os.path.join(self.results_directory, 'potential_errors.txt')
        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory)

    def write_errors(self, content):
        with open(self.errors_file, 'a') as f:
            f.write(content + '\n')

    def analyze_missing_values(self):
        missing_values = self.data.isnull().sum()
        unknown_values = (self.data == 'Unknown').sum()
        combined_report = missing_values + unknown_values
        self.write_errors("Missing or 'Unknown' Values:\n" + str(combined_report))
        print("Missing or 'Unknown' Values:\n", combined_report)

    def check_repeated_instances(self):
        repeated_instances = self.data[self.data.duplicated()]
        print("Repeated Instances:\n", repeated_instances)
        self.write_errors("Repeated Instances:\n" + str(repeated_instances))


class CatBreedAnalyzer:
    def __init__(self, file_path=None):

        # self.data is a dataframe
        if file_path is None:
            self.data = pd.read_excel('modified_dataset.xlsx')
        else:
            self.data = _remove_columns(file_path)
            self.analyzed_data = None

        self.results_directory = r"Catology\data\results"
        self.errors_file = os.path.join(self.results_directory, 'potential_errors.txt')
        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory)

    def write_errors(self, content):
        with open(self.errors_file, 'a') as f:
            f.write(content + '\n')

    def analyze_missing_values(self):
        missing_values = self.data.isnull().sum()
        unknown_values = (self.data == 'Unknown').sum()
        combined_report = missing_values + unknown_values
        self.write_errors("Missing or 'Unknown' Values:\n" + str(combined_report))
        print("Missing or 'Unknown' Values:\n", combined_report)

    def check_and_drop_repeated_instances(self):
        repeated_instances = self.data[self.data.duplicated()]
        print("Repeated Instances:\n", repeated_instances)
        self.write_errors("Repeated Instances:\n" + str(repeated_instances))
        print('Dropping them ...')
        self.data = self.data.drop_duplicates()

    def check_and_drop_unknown_instances(self):  # this one should run after _transform_abundance_column
        unknown_instances = self.data[self.data.isin(['Unknown']).any(axis=1)]
        print(f"Rows with 'Unknown' values:\n{unknown_instances}")
        self.data = self.data.drop(unknown_instances.index)
        self.save_to_excel()
        print(f"Dropped {len(unknown_instances)} rows containing 'Unknown' values.")

    # bar plot since the data is not numerical
    def count_and_show_instances_per_breed(self):
        breed_counts = self.data['Race Description'].value_counts()
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

        boxplot_file = os.path.join(results_dir, f'boxplot_breeds.png')

        # check if the files already exist
        if os.path.exists(boxplot_file):
            print(f"Images for the classification attributes already exist. Skipping saving for this column.")
            plt.close()  # Close the plot to free up memory

        plt.savefig(boxplot_file)
        plt.close()

    def plot_columns(self):
        # all columns except the last one
        columns_to_plot = self.data.columns[:-1]

        output_dir = results_dir
        os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

        for column in columns_to_plot:
            # some columns contain white blanks in their name
            safe_column_name = column.replace(" ", "_").replace("/", "_").replace("\\", "_")

            # convert columns to a numeric type (float), just in case
            try:
                self.data[column] = self.data[column].astype(float)
            except ValueError:
                print(f"Skipping non-numeric column: {column}")
                continue

            # Check if the column is numeric after conversion
            if not pd.api.types.is_numeric_dtype(self.data[column]):
                print(f"Skipping non-numeric column: {column}")
                continue

            plt.figure(figsize=(12, 5))

            # Histogram
            plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st plot
            plt.hist(self.data[column], bins=30, color='skyblue', edgecolor='black')
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')

            # Boxplot
            plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd plot
            plt.boxplot(self.data[column], vert=False)
            plt.title(f'Boxplot of {column}')
            plt.xlabel(column)

            plt.tight_layout()

            # save as images
            histogram_file = os.path.join(output_dir, f'histogram_{safe_column_name}.png')
            boxplot_file = os.path.join(output_dir, f'boxplot_{safe_column_name}.png')

            # check if the files already exist
            if os.path.exists(histogram_file) or os.path.exists(boxplot_file):
                print(f"Images for {column} already exist. Skipping saving for this column.")
                plt.close()  # Close the plot to free up memory
                continue

            plt.savefig(histogram_file)
            plt.savefig(boxplot_file)

            plt.close()  # Close the plot to free up memory

        print("Plots saved successfully in:", output_dir)

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
        self.data['Sex'] = self.data['Sex'].apply(map_sex)
        self.save_to_excel()

    def _transform_area_column(self):
        self.data['Urban/Rural area'] = self.data['Urban/Rural area'].apply(map_area)
        self.save_to_excel()

    def _transform_number_of_cats_column(self):
        self.data['Number of cats in the household'] = self.data['Number of cats in the household'].apply(map_cats)
        self.save_to_excel()

    def _transform_age_column(self):
        self.data['Age'] = self.data['Age'].apply(map_age)
        self.save_to_excel()

    def _transform_place_of_living(self):
        self.data['Place of living'] = self.data['Place of living'].apply(map_place_of_living)
        self.save_to_excel()

    """
    Factorizes the 'Race' column to get numerical and unique values
    Adds the numerical codes as a new column called 'Numerical Race'
    Renames the original 'Race' column to 'Race_Description' and keeps it
    Drops the original 'Race' column
    Reorders the columns so that 'Numerical Race' and 'Race_Description' are at the end
    """

    def _transform_abundance_column(self):

        file_path = path
        df = pd.read_excel(file_path)
        df['Abundance of natural areas'] = df['Abundance of natural areas'].replace('Unknown', np.nan)
        df['Abundance of natural areas'] = pd.to_numeric(df['Abundance of natural areas'], errors='coerce')
        median_value = int(df['Abundance of natural areas'].median())
        df['Abundance of natural areas'] = df['Abundance of natural areas'].fillna(median_value)
        df.to_excel(file_path, index=False)
        print(f"Replaced 'Unknown' values with the median: {median_value}")

    def _transform_race_column(self):

        if 'Numerical Race' in self.data.columns and 'Race Description' in self.data.columns:
            print("Transformation has already been applied. No further changes made.")
            return

        # Factorize 'Race' column to get numerical encoding and unique races
        numerical_race, unique_races = pd.factorize(self.data['Race'])

        # Add 'Numerical Race' column and rename 'Race' to 'Race Description'
        self.data['Numerical Race'] = numerical_race
        self.data['Race Description'] = self.data['Race']
        self.data = self.data.drop(columns=['Race'])

        # Reorder columns, placing 'Numerical Race' and 'Race Description' at the end
        other_columns = [col for col in self.data.columns if col not in ['Numerical Race', 'Race Description']]
        self.data = self.data[other_columns + ['Numerical Race', 'Race Description']]

        # Save the modified DataFrame to Excel
        self.save_to_excel()

    def add_numerical_consistency(self):
        # am nevoie si pentru place_of_living de transformat in valoare numerica
        # am facut place of living - Cosmin

        """
        self._transform_place_of_living()

        self._transform_abundance_column()

        self._transform_age_column()
        self._transform_sex_column()

        self._transform_area_column()
        self._transform_number_of_cats_column()

        # trebuie rulat si asta de jos de aici
        # self._transform_race_column()
        """
        self._transform_race_column()

    def save_to_excel(self, file=path):
        self.data.to_excel(file_name, index=False)

    # the correlation matrix still needs work
    def build_correlation_matrix(self):

        os.makedirs(results_dir, exist_ok=True)  # Create the directory if it doesn't exist

        # only numerical columns
        numerical_data = self.data.iloc[:, :-1]  # All columns except the last one

        # compute the correlation matrix
        correlation_matrix = numerical_data.corr()

        plt.figure(figsize=(12, 8))

        # heatmap with the correlation matrix
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})

        plt.title('Correlation Matrix')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)

        # Save the correlation matrix as an image
        correlation_matrix_file = os.path.join(results_dir, 'correlation_matrix.png')
        plt.savefig(correlation_matrix_file)
        plt.close()  # Close the plot to free up memory

        print("Correlation matrix saved successfully in:", results_dir)


if __name__ == "__main__":

    # the full relative path
    file_name = 'modified_dataset.xlsx'

    # get the full path based on the current working directory
    full_file_path = os.path.join(os.getcwd(), file_name)

    # instantiate the analyzer only if the file does not exist
    if not os.path.isfile(full_file_path):
        analyzer = OriginalDatasetAnalyzer(path)
        print("Analyzer successfully instantiated")
    else:
        print(f"{full_file_path} already exists. Analyzer not instantiated.")
        analyzer = CatBreedAnalyzer()

    # analyzer.analyze_missing_values()

    # analyzer.add_numerical_consistency()
    # analyzer.analyze_missing_values()
    # analyzer.check_and_drop_repeated_instances()
    # analyzer.check_and_drop_unknown_instances()  # - should be run after transform_abundance_column()

    # + boxplot-uri
    # analyzer.plot_columns()

    # analyzer.count_and_show_instances_per_breed()

    # analyzer.visualize_distribution('Age')

    analyzer.build_correlation_matrix()
