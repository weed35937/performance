# Student Performance Analytics Dashboard

An interactive Streamlit dashboard for analyzing and predicting student performance using machine learning.

## Features

- Interactive data visualization
- Real-time performance prediction
- Multi-class classification
- Key performance indicators
- Beautiful UI with responsive design

## Live Demo

Visit the live application at: [Your Streamlit App URL]

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
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Deploy your forked repository
5. Select the main file as `student_performance_app.py`

## Data

The application uses the UCI Student Performance Dataset. Make sure to place your data file (`student-mat.csv`) in the root directory.

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