import zipfile
import os

def extract_dataset():
    """Extract the student.zip file."""
    try:
        print("Extracting dataset...")
        with zipfile.ZipFile('student.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
        print("Dataset extracted successfully!")
        
        # Check if the file exists
        if os.path.exists('student-mat.csv'):
            print("Found student-mat.csv - ready for analysis!")
        else:
            print("Warning: student-mat.csv not found after extraction.")
            print("Please check the contents of the extracted files.")
            
    except Exception as e:
        print(f"Error extracting the dataset: {e}")

if __name__ == "__main__":
    extract_dataset() 