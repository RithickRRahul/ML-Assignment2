import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
import os

# Set page configuration
st.set_page_config(
    page_title="Indian Election Winner Prediction",
    page_icon="🗳️",
    layout="wide"
)

# App title and description
st.title("🗳️ Indian Election Winner Prediction")
st.markdown("""
This app uses machine learning to predict the likelihood of a candidate winning an election based on historical Indian election data.
Upload the dataset or use the sample data to see predictions and insights.
""")

# Function to load and preprocess data
@st.cache_data
def load_and_process_data(file):
    try:
        df = pd.read_csv(file)
        # Create Winner column if it doesn't exist
        if 'Winner' not in df.columns:
            df['Winner'] = df.groupby(['year', 'pc_no'])['totvotpoll'].transform(lambda x: x == x.max()).astype(int)
        
        # Handle missing values
        df['pc_type'] = df['pc_type'].fillna('UNKNOWN')
        df['cand_sex'] = df['cand_sex'].fillna('UNKNOWN')
        
        # Create vote share feature
        df['vote_share'] = df['totvotpoll'] / df['electors'] * 100
        
        return df
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        return None

# Sidebar for data upload and model selection
st.sidebar.header("📊 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload your election dataset (CSV)", type=['csv'])

if uploaded_file is not None:
    df = load_and_process_data(uploaded_file)
    st.sidebar.success("Data uploaded successfully!")
else:
    st.sidebar.info("Please upload a CSV file or use the sample data option below.")
    use_sample = st.sidebar.checkbox("Use sample data")
    if use_sample:
        # Replace with actual path or URL if you have sample data
        sample_path = "indian-national-level-election.csv"
        # Check if file exists (in real deployment, would need a default dataset)
        if os.path.exists(sample_path):
            df = load_and_process_data(sample_path)
        else:
            st.sidebar.warning("Sample data not found. Please upload your data.")
            df = None
    else:
        df = None

# Model selection
st.sidebar.header("🤖 Model Selection")
model_choice = st.sidebar.selectbox(
    "Select prediction model",
    ["Random Forest", "XGBoost"]
)

# Main app logic
if df is not None:
    # Display basic dataset info
    st.header("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", df.shape[0])
    with col2:
        st.metric("Winners", df['Winner'].sum())
    with col3:
        st.metric("Election Years", f"{df['year'].min()} - {df['year'].max()}")
    
    # Display sample data
    with st.expander("View Sample Data"):
        st.dataframe(df.head())
    
    # Feature Engineering and Analysis
    st.header("📈 Data Analysis")
    
    # Gender distribution
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Gender Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x='cand_sex', data=df)
        plt.title("Gender Distribution of Candidates")
        plt.xlabel("Gender")
        plt.ylabel("Number of Candidates")
        plt.grid(True)
        st.pyplot(fig)
        
    # Top winning parties
    with col2:
        st.subheader("Top Winning Parties")
        top_parties = df[df['Winner'] == 1]['partyabbre'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=top_parties.index, y=top_parties.values)
        plt.title("Top 10 Parties by Number of Winning Candidates")
        plt.xlabel("Party Abbreviation")
        plt.ylabel("Win Count")
        plt.xticks(rotation=45)
        plt.grid(True)
        st.pyplot(fig)
    
    # Yearly party performance
    st.subheader("Party Performance Over Time")
    
    # Get top 5 parties
    top_5_parties = df['partyabbre'].value_counts().head(5).index.tolist()
    selected_parties = st.multiselect(
        "Select parties to analyze",
        options=df['partyabbre'].unique().tolist(),
        default=top_5_parties
    )
    
    if selected_parties:
        # Filter for selected parties and winners
        party_data = df[df['partyabbre'].isin(selected_parties) & (df['Winner'] == 1)]
        
        # Create pivot table
        party_year = pd.crosstab(party_data['year'], party_data['partyabbre'])
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        party_year.plot(kind='bar', stacked=False, ax=ax)
        plt.title('Winning Seats by Party Over Years')
        plt.ylabel('Number of Constituencies Won')
        plt.xlabel('Election Year')
        plt.legend(title='Party')
        plt.grid(True, axis='y')
        st.pyplot(fig)

    # Prediction Section
    st.header("🔮 Election Winner Prediction")
    st.markdown("""
    Enter candidate and constituency details to predict the likelihood of winning.
    """)
    
    # Create form for prediction inputs
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            year = st.number_input("Election Year", min_value=int(df['year'].min()), max_value=2030, value=2024)
            party = st.selectbox("Party", options=df['partyabbre'].unique())
            state = st.selectbox("State", options=df['st_name'].unique())
            
        with col2:
            pc_type = st.selectbox("Constituency Type", options=df['pc_type'].unique())
            gender = st.selectbox("Candidate Gender", options=df['cand_sex'].unique())
            votes = st.number_input("Expected Votes", min_value=0, value=100000)
            
        with col3:
            electors = st.number_input("Total Electors", min_value=1, value=500000)
            vote_share = votes / electors * 100
            st.metric("Vote Share (%)", round(vote_share, 2))
            
        submit_button = st.form_submit_button("Predict")
        
    # Make prediction when form is submitted
    if submit_button:
        # Prepare data for prediction
        try:
            # Prepare sample data for encoding
            sample_data = pd.DataFrame({
                'year': [year],
                'totvotpoll': [votes],
                'electors': [electors],
                'vote_share': [vote_share],
                'st_name': [state],
                'pc_type': [pc_type],
                'cand_sex': [gender],
                'partyabbre': [party]
            })
            
            # Train encoders on full dataset
            cat_features = ['st_name', 'pc_type', 'cand_sex', 'partyabbre']
            df_encoded = pd.get_dummies(df[cat_features], drop_first=True)
            
            # Encode the prediction sample
            sample_encoded = pd.get_dummies(sample_data[cat_features], drop_first=True)
            
            # Align sample with training data columns
            missing_cols = set(df_encoded.columns) - set(sample_encoded.columns)
            for col in missing_cols:
                sample_encoded[col] = 0
            sample_encoded = sample_encoded[df_encoded.columns]
            
            # Combine with numerical features
            numerical_features = ['year', 'totvotpoll', 'electors', 'vote_share']
            X_sample = pd.concat([sample_data[numerical_features], sample_encoded], axis=1)
            
            # Train a quick model on full data (in production would load pre-trained model)
            features = ['year', 'totvotpoll', 'electors', 'vote_share']
            X = pd.concat([df[features], df_encoded], axis=1)
            y = df['Winner']
            
            if model_choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42)
                
            with st.spinner("Training model..."):
                model.fit(X, y)
            
            # Make prediction
            prediction_prob = model.predict_proba(X_sample)[0][1]
            prediction = model.predict(X_sample)[0]
            
            # Display prediction
            st.subheader("Prediction Result")
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.success(f"This candidate is likely to win with {prediction_prob:.2%} probability")
                else:
                    st.error(f"This candidate is unlikely to win with only {prediction_prob:.2%} probability")
                    
            with col2:
                # Create gauge chart for win probability
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.set_xlim(0, 10)
                ax.set_ylim(0, 10)
                ax.add_patch(plt.Circle((5, 5), 4.5, fc='#f0f0f0'))
                ax.add_patch(plt.Circle((5, 5), 4, fc='white'))
                
                # Create colored arc for probability
                theta1 = -90  # Starting at top
                theta2 = theta1 + 360 * prediction_prob
                ax.add_patch(plt.matplotlib.patches.Wedge((5, 5), 4.5, theta1, theta2, fc='#1f77b4'))
                
                # Add text
                ax.text(5, 5, f"{prediction_prob:.1%}", ha='center', va='center', fontsize=24)
                ax.text(5, 3.5, "Win Probability", ha='center', fontsize=10)
                
                # Remove axes
                ax.set_axis_off()
                st.pyplot(fig)
            
            # Show likely winning factors
            if hasattr(model, 'feature_importances_'):
                st.subheader("Key Factors Influencing the Prediction")
                feature_importance = model.feature_importances_
                feature_names = X.columns
                
                # Get top 10 important features
                indices = np.argsort(feature_importance)[-10:]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(range(len(indices)), feature_importance[indices], align="center")
                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels([feature_names[i] for i in indices])
                ax.set_xlabel("Feature Importance")
                ax.set_title("Top 10 Influential Factors")
                st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            
    # Historical insights
    st.header("📜 Historical Insights")
    
    # Party win rate comparison
    st.subheader("Party Win Rates")
    
    # Calculate win rates by party
    party_stats = df.groupby('partyabbre').agg(
        contests=('Winner', 'count'),
        wins=('Winner', 'sum')
    )
    party_stats['win_rate'] = party_stats['wins'] / party_stats['contests'] * 100
    party_stats = party_stats.sort_values('win_rate', ascending=False)
    
    # Filter to parties with at least 50 contests
    major_parties = party_stats[party_stats['contests'] >= 50].head(10)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=major_parties.index, y=major_parties['win_rate'], palette='viridis')
    plt.title('Win Rate by Major Political Parties (min. 50 contests)')
    plt.xlabel('Party')
    plt.ylabel('Win Rate (%)')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    st.pyplot(fig)
    
    # Add information about most likely winners
    st.subheader("Most Likely Winners Based on Historical Data")
    
    # Find parties with highest win rates in recent elections
    latest_year = df['year'].max()
    recent_df = df[df['year'] >= latest_year - 10]
    
    recent_wins = recent_df.groupby('partyabbre').agg(
        contests=('Winner', 'count'),
        wins=('Winner', 'sum')
    )
    recent_wins['win_rate'] = recent_wins['wins'] / recent_wins['contests'] * 100
    recent_wins = recent_wins[recent_wins['contests'] >= 20].sort_values('win_rate', ascending=False)
    
    st.write(f"""
    Based on historical data analysis of Indian elections, the parties with the highest win rates in recent elections are:
    
    1. **{recent_wins.index[0]}** - {recent_wins['win_rate'].iloc[0]:.1f}% win rate ({int(recent_wins['wins'].iloc[0])} wins out of {int(recent_wins['contests'].iloc[0])} contests)
    2. **{recent_wins.index[1]}** - {recent_wins['win_rate'].iloc[1]:.1f}% win rate ({int(recent_wins['wins'].iloc[1])} wins out of {int(recent_wins['contests'].iloc[1])} contests)
    3. **{recent_wins.index[2]}** - {recent_wins['win_rate'].iloc[2]:.1f}% win rate ({int(recent_wins['wins'].iloc[2])} wins out of {int(recent_wins['contests'].iloc[2])} contests)
    """)
    
    st.info("""
    **Note:** Past performance does not guarantee future results. Election outcomes are influenced by many factors including:
    - Current political climate
    - Candidate popularity
    - Regional issues
    - Voter turnout
    - Campaign effectiveness
    """)

else:
    # Display when no data is available
    st.info("Please upload a dataset or use the sample data option to get started.")
    
    # Show example of expected data format
    st.subheader("Expected Data Format")
    example_data = pd.DataFrame({
        'year': [2019, 2019, 2019, 2019],
        'pc_no': [1, 1, 2, 2],
        'st_name': ['State1', 'State1', 'State2', 'State2'],
        'pc_type': ['GEN', 'GEN', 'SC', 'SC'],
        'partyabbre': ['BJP', 'INC', 'BJP', 'INC'],
        'cand_sex': ['M', 'F', 'M', 'F'],
        'totvotpoll': [150000, 120000, 180000, 200000],
        'electors': [400000, 400000, 500000, 500000]
    })
    st.dataframe(example_data)

# Add footer
st.markdown("---")
st.markdown("Election Winner Prediction App | Data Science Project")