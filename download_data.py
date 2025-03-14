import urllib.request
import os

def download_dataset():
    """Download the Student Performance Dataset from UCI ML Repository."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
    print("Downloading dataset...")
    
    try:
        # Download the zip file
        urllib.request.urlretrieve(url, "student.zip")
        print("Dataset downloaded successfully!")
        
        # Unzip the file (you'll need to unzip it manually as it's a zip file)
        print("\nPlease unzip the student.zip file and ensure student-mat.csv is in the current directory.")
        print("The file student-mat.csv contains the Mathematics course data we'll be using.")
        
    except Exception as e:
        print(f"Error downloading the dataset: {e}")
        print("\nPlease manually download the dataset from:")
        print("https://archive.ics.uci.edu/ml/datasets/Student+Performance")

if __name__ == "__main__":
    download_dataset() 