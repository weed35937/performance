# [Student Performance Analytics Dashboard](https://student-perform-analysis.streamlit.app/)

An interactive Streamlit dashboard for analyzing and predicting student performance using machine learning.

## Features

- Interactive data visualization
- Real-time performance prediction
- Multi-class classification
- Key performance indicators
- Beautiful UI with responsive design

## Data Setup (Required)

Before running the application, you need to download the dataset:

1. Download the dataset:
   - Visit [UCI Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance)
   - Or use direct link: [student-mat.csv](https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student-mat.csv)
2. Place the downloaded `student-mat.csv` file in the `data` directory of the project

## Local Development

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [your-repo-name]
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run student_performance_app.py
```

The application will be available at http://localhost:8501

## Deployment

This application is ready to be deployed on Streamlit Community Cloud:

1. Fork this repository
2. Download the dataset and place `student-mat.csv` in the `data` directory
3. Commit and push the changes including the data file
4. Go to [share.streamlit.io](https://share.streamlit.io)
5. Sign in with GitHub
6. Deploy your forked repository
7. Select the main file as `student_performance_app.py`

## Project Structure
```
.
├── data/
│   └── student-mat.csv    # Dataset file (you need to download this)
├── student_performance_app.py
├── student_performance_analysis.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Requirements

All requirements are listed in `requirements.txt`. The main dependencies are:
- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
