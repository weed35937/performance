import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Data Loading and Preprocessing
def load_data(file_path):
    """Load and prepare the student performance dataset."""
    # Read the data
    df = pd.read_csv(file_path, sep=';')
    
    # Create binary target variable
    df['performance_binary'] = (df['G3'] >= 15).astype(int)
    
    # Create multi-class target variable
    df['performance_class'] = pd.qcut(df['G3'], q=4, labels=['Poor', 'Fair', 'Good', 'Excellent'])
    
    return df

def visualize_data(df):
    """Create visualizations for data analysis."""
    # Set style parameters
    plt.style.use('default')
    sns.set_theme(style="whitegrid")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Grade Distribution
    plt.subplot(2, 2, 1)
    sns.histplot(data=df, x='G3', bins=20)
    plt.title('Distribution of Final Grades')
    plt.xlabel('Final Grade')
    plt.ylabel('Count')
    
    # 2. Correlation between grades
    plt.subplot(2, 2, 2)
    grade_corr = df[['G1', 'G2', 'G3']].corr()
    sns.heatmap(grade_corr, annot=True, cmap='coolwarm')
    plt.title('Correlation between Period Grades')
    
    # 3. Study Time vs Final Grade
    plt.subplot(2, 2, 3)
    sns.boxplot(data=df, x='studytime', y='G3')
    plt.title('Study Time vs Final Grade')
    plt.xlabel('Study Time (1-4 scale)')
    plt.ylabel('Final Grade')
    
    # 4. Absences vs Performance
    plt.subplot(2, 2, 4)
    sns.boxplot(data=df, x='performance_binary', y='absences')
    plt.title('Absences vs Performance')
    plt.xlabel('Good Performance (0=No, 1=Yes)')
    plt.ylabel('Number of Absences')
    
    plt.tight_layout()
    plt.savefig('performance_analysis.png')
    plt.close()

def prepare_features(df, target='binary'):
    """Prepare features for modeling."""
    # Select relevant features
    feature_columns = [
        'age', 'studytime', 'failures', 'absences',
        'G1', 'G2',  # First and second period grades
        'Medu', 'Fedu',  # Parent's education
        'traveltime', 'studytime',
        'freetime', 'goout', 'health'
    ]
    
    # Convert categorical variables to dummy variables
    categorical_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 
                         'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 
                         'famsup', 'paid', 'activities', 'nursery', 'higher', 
                         'internet', 'romantic']
    
    # Create dummy variables
    df_encoded = pd.get_dummies(df, columns=categorical_columns)
    
    # Combine numerical and encoded categorical features
    X = df_encoded[feature_columns + [col for col in df_encoded.columns 
                                    if any(cat in col for cat in categorical_columns)]]
    
    # Select target variable based on classification type
    if target == 'binary':
        y = df_encoded['performance_binary']
    else:
        y = df_encoded['performance_class']
    
    return X, y

def train_model(X, y, model_type='binary'):
    """Train a Random Forest model."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced' if model_type == 'multiclass' else None
    )
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    return model, X_train_scaled, X_test_scaled, y_train, y_test, y_pred

def analyze_results(model, X, y_test, y_pred, feature_names):
    """Analyze and visualize the results."""
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Visualize feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
    plt.title('Top 15 Most Important Features')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return feature_importance

def main():
    # Load the data
    print("Loading data...")
    df = load_data('student-mat.csv')
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualize_data(df)
    
    # Binary Classification
    print("\nPerforming binary classification...")
    X_binary, y_binary = prepare_features(df, target='binary')
    model_binary, *binary_results = train_model(X_binary, y_binary, 'binary')
    feature_importance_binary = analyze_results(model_binary, X_binary, *binary_results[-2:], X_binary.columns)
    
    # Multi-class Classification
    print("\nPerforming multi-class classification...")
    X_multi, y_multi = prepare_features(df, target='multiclass')
    model_multi, *multi_results = train_model(X_multi, y_multi, 'multiclass')
    feature_importance_multi = analyze_results(model_multi, X_multi, *multi_results[-2:], X_multi.columns)
    
    print("\nAnalysis complete! Visualizations have been saved to:")
    print("- performance_analysis.png")
    print("- feature_importance.png")
    print("- confusion_matrix.png")

if __name__ == "__main__":
    main()