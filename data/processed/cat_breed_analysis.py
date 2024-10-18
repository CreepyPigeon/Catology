import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from data_processing import clean_data

load_dotenv()
path = os.getenv('FILE_PATH')
results_dir = os.getenv('RESULTS_DIR')
errors_file = os.getenv('ERRORS_FILE')
new_data = os.getenv('NEW_DATA')

class CatBreedAnalyzer:
    def __init__(self, original, new_dataset=None):

        if new_dataset and os.path.isfile(new_dataset):
            # it exists, load the new dataset
            print(f"Loading existing dataset: {new_dataset}")
            self.data = pd.read_excel(new_dataset)
        else:
            # new_dataset doesn't exist, clean the original dataset and save it to the new dataset
            print(f"Cleaning original dataset: {original}")
            self.data = clean_data(original)

            # save the cleaned dataset to the new dataset file
            if new_dataset:
                self.data.to_excel(new_dataset, index=False)
                print(f"Cleaned dataset saved as: {new_dataset}")

        self.results_directory = results_dir
        self.errors_file = errors_file
        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory)

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
        print('Successfully saved the boxplot of cat breeds')
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

    # trebuie si asta pusa in directory-ul de rezultate
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

    # the correlation matrix still needs work
    def build_correlation_matrix(self):

        os.makedirs(results_dir, exist_ok=True)  # Create the directory if it doesn't exist

        # Select only numerical columns (e.g., float, int) and exclude non-numerical columns like 'Race Description'
        numerical_data = self.data.select_dtypes(include=[float, int])

        # Check if any numerical columns are available
        if numerical_data.empty:
            print("No numerical columns found to compute correlation matrix.")
            return

        # Compute the correlation matrix
        correlation_matrix = numerical_data.corr()

        plt.figure(figsize=(12, 8))

        # Heatmap with the correlation matrix
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
    # mai trebuie testat ca am push pasii in ordinea corecta si ca totul merge

    original_dataset = 'Translated_Cat_Dataset.xlsx'
    modified_dataset = new_data

    # get the full path based on the current working directory
    full_file_path = os.path.join(os.getcwd(), original_dataset)

    # instantiate the analyzer only if the file does not exist
    if not os.path.isfile(full_file_path):
        analyzer = CatBreedAnalyzer(full_file_path)
        print("Analyzer successfully instantiated")
    else:
        print(f"{full_file_path} already exists. Analyzer not instantiated.")
        analyzer = CatBreedAnalyzer(full_file_path, modified_dataset)

    analyzer.build_correlation_matrix()
    analyzer.plot_columns()

    print('Now showing instances per breed')
    analyzer.count_and_show_instances_per_breed()
