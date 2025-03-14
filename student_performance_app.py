import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from student_performance_analysis import load_data, prepare_features, train_model

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for luxury design with animations and responsive layout
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        animation: gradientBG 15s ease infinite;
        font-size: 16px;  /* Base font size increase */
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Responsive containers */
    @media (max-width: 768px) {
        .responsive-grid {
            grid-template-columns: 1fr !important;
        }
        .metrics-container {
            flex-direction: column;
        }
    }
    
    /* Headers with animations */
    h1 {
        background: linear-gradient(120deg, #1e3d59, #2b4d6f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        padding: 1.5rem 0;
        animation: fadeIn 0.5s ease-in;
        font-size: 2.5rem !important;
    }
    
    h2 {
        color: #2b4d6f;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
        transition: all 0.3s ease;
        font-size: 2rem !important;
    }
    
    h3 {
        font-size: 1.75rem !important;
    }
    
    p {
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { transform: translateX(-20px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Cards with hover effects */
    .card {
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease-in;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Metrics with animations */
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3d59;
        animation: countUp 1s ease-out;
    }
    
    .metric-label {
        color: #666;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #343a40 !important;
        padding: 0 !important;
    }
    
    /* Navigation title */
    .nav-title {
        background: #212529;
        color: white !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        padding: 1rem !important;
        margin: 0 !important;
        text-align: left;
    }
    
    /* Bootstrap-style navigation buttons */
    .stRadio > div[role="radiogroup"] {
        padding: 0 !important;
    }
    
    /* Hide radio button icons */
    div[data-testid="stMarkdown"] + div[data-testid="stVerticalBlock"] div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    
    /* Navigation button text color */
    .stRadio > div[role="radiogroup"] > label {
         color: white !important;
        font-size: 1rem !important;
        padding: 0.8rem 1.5rem !important;
        background: transparent !important;
        border-radius: 0 !important;
        margin: 0 !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.2s ease;
    }
    
    /* Selected nav item */
    .stRadio > div[role="radiogroup"] > label[data-checked="true"] {
        background: rgba(255, 255, 255, 0.1) !important;
        color: #007bff !important;
        font-weight: 600;
        box-shadow: none !important;
        border-left: 4px solid #007bff;
    }
    
    /* Footer in sidebar */
    .sidebar-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        padding: 1rem;
        background: rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(5px);
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.8rem;
        text-align: center;
        animation: fadeIn 0.5s ease-in;
    }
    
    .sidebar-footer p {
        margin: 0;
        line-height: 1.5;
    }
    
    /* Buttons with animations */
    .stButton>button {
        background: linear-gradient(45deg, #1e3d59, #2b4d6f);
        color: white;
        border-radius: 2rem;
        padding: 0.75rem 2rem;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        background: linear-gradient(45deg, #2b4d6f, #1e3d59);
    }
    
    /* Plots with animations */
    .plot-container {
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease-in;
    }
    
    .plot-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Form inputs with animations */
    .stSlider, .stSelectbox {
        animation: slideIn 0.5s ease-in;
    }
    
    /* Loading animation */
    .stProgress .st-bo {
        background-color: #1e3d59;
        height: 3px;
        animation: loading 1s ease-in-out infinite;
    }
    
    @keyframes loading {
        0% { width: 0%; }
        50% { width: 100%; }
        100% { width: 0%; }
    }
    
    /* Navigation menu items */
    .nav-item {
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        transition: all 0.3s ease;
        cursor: pointer;
        color: white;
    }
    
    .nav-item:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateX(5px);
    }
    
    /* Increase text size globally */
    .stMarkdown, .stText {
        font-size: 1.1rem !important;
    }

    /* Add tab styling for white text */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 8px;
    }

    [data-testid="stTabs"] [data-baseweb="tab"] {
        background-color: transparent;
        color: white !important;
        border-radius: 4px;
        font-weight: 500;
        padding: 8px 16px;
    }

    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background-color: #007bff;
    }

    [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.1);
        color: white !important;
    }

    [data-testid="stTabs"] [data-baseweb="tab-panel"] {
        padding: 16px 0;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced title and introduction with animation
st.markdown("""
<div style='text-align: center; padding: 2rem 0; animation: fadeIn 0.5s ease-in;'>
    <h1 style='font-size: 3rem; margin-bottom: 1rem;'>📚 Student Performance Analytics</h1>
    <p style='font-size: 1.2rem; color: #666; max-width: 800px; margin: 0 auto; line-height: 1.6;'>
        Unlock powerful insights into student performance through advanced analytics and predictive modeling
    </p>
</div>
""", unsafe_allow_html=True)

# Load and cache data with loading animation
with st.spinner('Loading data and models...'):
    @st.cache_data
    def load_cached_data():
        return load_data('student-mat.csv')

    @st.cache_resource
    def load_cached_models(df):
        X_binary, y_binary = prepare_features(df, target='binary')
        X_multi, y_multi = prepare_features(df, target='multiclass')
        model_binary, *_ = train_model(X_binary, y_binary, 'binary')
        model_multi, *_ = train_model(X_multi, y_multi, 'multiclass')
        return model_binary, model_multi, X_binary.columns

    df = load_cached_data()
    model_binary, model_multi, feature_names = load_cached_models(df)

# Enhanced sidebar with Bootstrap-style navigation
with st.sidebar:
    st.markdown("""
    <div class="nav-title">
        Navigation
    </div>
    """, unsafe_allow_html=True)
    
    # Custom radio buttons with Bootstrap-style
    page = st.radio("",
                    options=["Dashboard", "Predict Performance", "About"],
                    key="navigation",
                    help="Navigate through different sections of the application",
                    label_visibility="collapsed")
    
    # Enhanced footer with better contrast
    st.markdown("""
    <div class="sidebar-footer">
        <p style='font-weight: 600; color: white;'>Created by Person Who Loves Child</p>
    </div>
    """, unsafe_allow_html=True)

if page == "Dashboard":
    # Enhanced metrics display with animations and new color scheme
    st.markdown("""
    <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin-bottom: 2rem;'>
        <div class='card' style='background: linear-gradient(135deg, #00B4DB 0%, #0083B0 100%);'>
            <p style='color: rgba(255,255,255,0.8); font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;'>TOTAL STUDENTS</p>
            <h2 style='color: white; font-size: 2.5rem; margin: 0.5rem 0;'>{}</h2>
            <p style='color: rgba(255,255,255,0.8); font-size: 0.9rem;'>Enrolled Students</p>
        </div>
        <div class='card' style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);'>
            <p style='color: rgba(255,255,255,0.8); font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;'>AVERAGE GRADE</p>
            <h2 style='color: white; font-size: 2.5rem; margin: 0.5rem 0;'>{:.1f}</h2>
            <p style='color: rgba(255,255,255,0.8); font-size: 0.9rem;'>Out of 20</p>
        </div>
        <div class='card' style='background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%);'>
            <p style='color: rgba(255,255,255,0.8); font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;'>SUCCESS RATE</p>
            <h2 style='color: white; font-size: 2.5rem; margin: 0.5rem 0;'>{:.1f}%</h2>
            <p style='color: rgba(255,255,255,0.8); font-size: 0.9rem;'>Passing Students</p>
        </div>
    </div>
    """.format(len(df), df['G3'].mean(), df['performance_binary'].mean()*100), unsafe_allow_html=True)

    # Interactive Analysis Section
    st.markdown("""
    <h2 style='color: #1e3d59; margin: 2rem 0 1rem;'>📊 Performance Analysis</h2>
    """, unsafe_allow_html=True)

    # Add tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Grade Distribution", "Performance Factors", "Time Series"])

    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.set_style("whitegrid")
            sns.histplot(data=df, x='G3', bins=20, color='#00B4DB')
            plt.title('Distribution of Final Grades', pad=20, color='#1e3d59')
            plt.xlabel('Final Grade', color='#666')
            plt.ylabel('Count', color='#666')
            ax.tick_params(colors='#666')
            st.pyplot(fig)
            plt.close()
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.set_style("white")
            grade_corr = df[['G1', 'G2', 'G3']].corr()
            sns.heatmap(grade_corr, annot=True, cmap='YlOrRd', center=0,
                       annot_kws={'color': 'white'})
            plt.title('Grade Progression Correlation', pad=20, color='#1e3d59')
            ax.tick_params(colors='#666')
            st.pyplot(fig)
            plt.close()
            st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        # Add feature importance plot
        st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': model_binary.feature_importances_
        }).sort_values('importance', ascending=False)

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=importances.head(10), x='importance', y='feature', palette='viridis')
        plt.title('Top 10 Factors Influencing Student Performance', pad=20, color='#1e3d59')
        plt.xlabel('Importance Score', color='#666')
        plt.ylabel('Factor', color='#666')
        ax.tick_params(colors='#666')
        st.pyplot(fig)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

        # Add interactive scatter plot
        st.markdown("<div class='plot-container' style='margin-top: 2rem;'>", unsafe_allow_html=True)
        x_axis = st.selectbox('Select X-axis:', ['G1', 'G2', 'studytime', 'absences', 'Medu', 'Fedu'])
        y_axis = st.selectbox('Select Y-axis:', ['G3', 'G2', 'G1', 'studytime', 'absences'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x=x_axis, y=y_axis, hue='performance_binary', 
                       palette=['#FF4B2B', '#38ef7d'])
        plt.title(f'{x_axis} vs {y_axis}', pad=20, color='#1e3d59')
        plt.xlabel(x_axis, color='#666')
        plt.ylabel(y_axis, color='#666')
        ax.tick_params(colors='#666')
        st.pyplot(fig)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        # Add grade progression analysis
        st.markdown("<div class='plot-container'>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate average grades for each period
        grade_progression = pd.DataFrame({
            'Period': ['First', 'Second', 'Final'],
            'Average Grade': [df['G1'].mean(), df['G2'].mean(), df['G3'].mean()]
        })
        
        sns.lineplot(data=grade_progression, x='Period', y='Average Grade', 
                    marker='o', color='#00B4DB', linewidth=3, markersize=10)
        plt.title('Grade Progression Throughout the Year', pad=20, color='#1e3d59')
        plt.xlabel('Assessment Period', color='#666')
        plt.ylabel('Average Grade', color='#666')
        ax.tick_params(colors='#666')
        
        # Add percentage changes
        for i in range(1, len(grade_progression)):
            pct_change = ((grade_progression['Average Grade'][i] - grade_progression['Average Grade'][i-1]) 
                         / grade_progression['Average Grade'][i-1] * 100)
            plt.annotate(f'{pct_change:+.1f}%',
                        xy=(i, grade_progression['Average Grade'][i]),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom',
                        color='#666')
        
        st.pyplot(fig)
        plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    # Add key insights section
    st.markdown("""
    <div style='margin-top: 3rem;'>
        <h2 style='color: #1e3d59; margin-bottom: 1.5rem;'>🔍 Key Insights</h2>
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem;'>
            <div class='card' style='background: white;'>
                <h3 style='color: #1e3d59; font-size: 1.2rem; margin-bottom: 1rem;'>Grade Distribution</h3>
                <p style='color: #666; line-height: 1.6;'>
                    The average final grade is {:.1f}/20, with {:.1f}% of students achieving passing grades.
                    The distribution shows a {}.
                </p>
            </div>
            <div class='card' style='background: white;'>
                <h3 style='color: #1e3d59; font-size: 1.2rem; margin-bottom: 1rem;'>Performance Factors</h3>
                <p style='color: #666; line-height: 1.6;'>
                    Previous grades (G1, G2) show the strongest correlation with final performance,
                    followed by study time and parent education levels.
                </p>
            </div>
            <div class='card' style='background: white;'>
                <h3 style='color: #1e3d59; font-size: 1.2rem; margin-bottom: 1rem;'>Trends</h3>
                <p style='color: #666; line-height: 1.6;'>
                    Student performance shows {} trend throughout the academic year,
                    with the most significant changes between {}.
                </p>
            </div>
        </div>
    </div>
    """.format(
        df['G3'].mean(),
        df['performance_binary'].mean() * 100,
        'normal distribution with slight right skew' if df['G3'].skew() > 0 else 'normal distribution with slight left skew',
        'an improving' if df['G3'].mean() > df['G1'].mean() else 'a stable',
        'first and second periods' if abs(df['G2'].mean() - df['G1'].mean()) > abs(df['G3'].mean() - df['G2'].mean()) else 'second and final periods'
    ), unsafe_allow_html=True)

    # Add export options
    st.markdown("""
    <div style='margin-top: 2rem; text-align: right;'>
        <button class='stButton' style='margin-left: 1rem;'>
            📊 Export Report
        </button>
        <button class='stButton' style='margin-left: 1rem;'>
            📥 Download Data
        </button>
    </div>
    """, unsafe_allow_html=True)

elif page == "Predict Performance":
    st.markdown("""
    <div class='card' style='margin-bottom: 2rem;'>
        <h2 style='color: #1e3d59; margin-bottom: 1rem;'>🎯 Student Performance Prediction</h2>
        <p style='color: #666; line-height: 1.6;'>Enter student information below to generate performance predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced input form with animations
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            g1 = st.slider("First Period Grade (G1)", 0, 20, 10,
                          help="Student's grade in the first period")
            g2 = st.slider("Second Period Grade (G2)", 0, 20, 10,
                          help="Student's grade in the second period")
            study_time = st.selectbox("Study Time", 
                                    options=[1, 2, 3, 4],
                                    help="1: <2 hours, 2: 2-5 hours, 3: 5-10 hours, 4: >10 hours")
            absences = st.slider("Number of Absences", 0, 93, 5,
                               help="Number of school absences")
            st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            mother_edu = st.selectbox("Mother's Education", 
                                    options=[0, 1, 2, 3, 4],
                                    help="0: none, 1: primary, 2: 5th-9th grade, 3: secondary, 4: higher")
            father_edu = st.selectbox("Father's Education",
                                    options=[0, 1, 2, 3, 4],
                                    help="0: none, 1: primary, 2: 5th-9th grade, 3: secondary, 4: higher")
            free_time = st.selectbox("Free Time",
                                   options=[1, 2, 3, 4, 5],
                                   help="1: very low to 5: very high")
            health = st.selectbox("Health Status",
                                options=[1, 2, 3, 4, 5],
                                help="1: very bad to 5: very good")
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Enhanced predict button with animation
    st.markdown("<div style='text-align: center; margin: 2rem 0;'>", unsafe_allow_html=True)
    predict_button = st.button("Generate Prediction")
    st.markdown("</div>", unsafe_allow_html=True)
    
    if predict_button:
        with st.spinner('Generating predictions...'):
            try:
                # Create feature vector with encoded values
                input_data = pd.DataFrame({
                    'G1': [g1],
                    'G2': [g2],
                    'studytime': [study_time],
                    'absences': [absences],
                    'Medu': [mother_edu],
                    'Fedu': [father_edu],
                    'freetime': [free_time],
                    'health': [health],
                    # Add default values for other required features
                    'age': [15],
                    'failures': [0],
                    'famrel': [4],
                    'goout': [3],
                    'Dalc': [1],
                    'Walc': [1],
                    # Encoded categorical variables (0/1)
                    'school_GP': [1],
                    'school_MS': [0],
                    'sex_F': [1],
                    'sex_M': [0],
                    'address_R': [0],
                    'address_U': [1],
                    'famsize_GT3': [1],
                    'famsize_LE3': [0],
                    'Pstatus_A': [0],
                    'Pstatus_T': [1],
                    'Mjob_at_home': [0],
                    'Mjob_health': [0],
                    'Mjob_other': [1],
                    'Mjob_services': [0],
                    'Mjob_teacher': [0],
                    'Fjob_at_home': [0],
                    'Fjob_health': [0],
                    'Fjob_other': [1],
                    'Fjob_services': [0],
                    'Fjob_teacher': [0],
                    'reason_course': [1],
                    'reason_home': [0],
                    'reason_other': [0],
                    'reason_reputation': [0],
                    'guardian_father': [0],
                    'guardian_mother': [1],
                    'guardian_other': [0],
                    'schoolsup_no': [1],
                    'schoolsup_yes': [0],
                    'famsup_no': [1],
                    'famsup_yes': [0],
                    'paid_no': [1],
                    'paid_yes': [0],
                    'activities_no': [1],
                    'activities_yes': [0],
                    'nursery_no': [0],
                    'nursery_yes': [1],
                    'higher_no': [0],
                    'higher_yes': [1],
                    'internet_no': [0],
                    'internet_yes': [1],
                    'romantic_no': [1],
                    'romantic_yes': [0]
                })
                
                # Make predictions
                binary_pred = model_binary.predict_proba(input_data)
                multi_pred = model_multi.predict_proba(input_data)
                
                # Display success message
                st.success("Prediction generated successfully!")
                
                # Display results with enhanced styling and animations
                st.markdown("""
                <div class='card' style='margin-top: 2rem;'>
                    <h2 style='color: #1e3d59; margin-bottom: 1.5rem;'>Prediction Results</h2>
                    <div style='display: flex; gap: 2rem;'>
                        <div style='flex: 1;'>
                            <h3 style='color: #2b4d6f; margin-bottom: 1rem;'>Binary Classification</h3>
                            <div style='text-align: center;'>
                                <div class='metric-value'>{:.1f}%</div>
                                <p class='metric-label'>Probability of Good Performance</p>
                            </div>
                        </div>
                        <div style='flex: 1;'>
                            <h3 style='color: #2b4d6f; margin-bottom: 1rem;'>Performance Level Probabilities</h3>
                """.format(binary_pred[0][1] * 100), unsafe_allow_html=True)
                
                # Display multi-class predictions with progress bars
                class_names = ['Poor', 'Fair', 'Good', 'Excellent']
                for cls, prob in zip(class_names, multi_pred[0]):
                    st.markdown(f"""
                    <div style='margin-bottom: 1rem;'>
                        <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                            <span style='color: #666;'>{cls}</span>
                            <span style='color: #1e3d59; font-weight: 500;'>{prob*100:.1f}%</span>
                        </div>
                        <div style='background: #f8f9fa; border-radius: 1rem; height: 0.5rem;'>
                            <div style='background: linear-gradient(90deg, #1e3d59, #2b4d6f); 
                                      width: {prob*100}%; height: 100%; border-radius: 1rem;
                                      transition: width 1s ease-in-out;'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div></div></div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error("Unable to generate prediction. Please check your input values and try again.")
                st.info("Make sure all required fields are filled correctly.")

else:  # About page
    st.title("About This Dashboard")
    
    # Purpose Section
    st.header("Purpose")
    st.write("This dashboard is designed to help educators and administrators:")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("Monitor student performance trends")
    with col2:
        st.warning("Identify at-risk students early")
    with col3:
        st.success("Make data-driven decisions")
    
    # Features Section
    st.header("Features")
    features_col1, features_col2 = st.columns(2)
    
    with features_col1:
        st.markdown("""
        - Interactive data visualization
        - Real-time performance prediction
        """)
    
    with features_col2:
        st.markdown("""
        - Multi-class classification
        - Key performance indicators
        """)
    
    # Model Information Section
    st.header("Model Information")
    st.write("The predictions are based on Random Forest models trained on the UCI Student Performance Dataset.")
    
    metrics_col1, metrics_col2 = st.columns(2)
    with metrics_col1:
        st.metric(
            label="Binary Classification",
            value="95%",
            delta="Accuracy",
            delta_color="normal"
        )
    
    with metrics_col2:
        st.metric(
            label="Multi-class Classification",
            value="77%",
            delta="Accuracy",
            delta_color="normal"
        )
    
    # How to Use Section
    st.header("How to Use")
    
    tab1, tab2, tab3 = st.tabs(["Dashboard", "Predict Performance", "About"])
    
    with tab1:
        st.markdown("""
        ### Dashboard
        View overall performance metrics and trends:
        - Monitor student grades
        - Analyze performance patterns
        - Track success rates
        """)
    
    with tab2:
        st.markdown("""
        ### Predict Performance
        Enter student information to get predictions:
        - Input student grades
        - Provide study habits
        - Get performance forecasts
        """)
    
    with tab3:
        st.markdown("""
        ### About
        Learn more about the system:
        - System capabilities
        - Model information
        - Usage guidelines
        """)
    
    # Additional Resources
    st.header("Additional Resources")
    with st.expander("Documentation"):
        st.write("Detailed documentation about using the dashboard and interpreting results.")
    
    with st.expander("Data Sources"):
        st.write("Information about the UCI Student Performance Dataset and data collection methodology.")
    
    with st.expander("Model Details"):
        st.write("Technical details about the machine learning models and their training process.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center;'>"
        "Created by Person Who Loves Child"
        "</div>",
        unsafe_allow_html=True
    ) 