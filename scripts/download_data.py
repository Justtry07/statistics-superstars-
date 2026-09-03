from src.data_loader import DataLoader
import pandas as pd

# Initialize loader
loader = DataLoader()

# Option 1: Iris Dataset
def load_iris():
    """Load Iris dataset from scikit-learn."""
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target_names[iris.target]
    return df

# Option 2: Student Performance
def load_student_performance():
    """Load Student Performance dataset from UCI."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
    # Note: This needs unzipping - alternative: download manually
    return None

# Option 3: Temperature
def load_temperature():
    """Load NASA temperature data."""
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
    return loader.load_from_url(url, "temperature.csv")

# Option 4: COVID-19
def load_covid():
    """Load COVID-19 data from Our World in Data."""
    url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    return loader.load_from_url(url, "covid_data.csv")

# Choose your dataset
df = load_iris()  

# Save raw data
loader.save_processed_data(df, "raw_data.csv")
print(f"Dataset loaded: {df.shape}")
