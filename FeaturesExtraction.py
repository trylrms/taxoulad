import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Import StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np
import time


def load_data(student_vle_file, vle_file, assessments_file, student_assessment_file):
    """Loads data files into pandas DataFrames."""
    try:
      student_vle_data_df = pd.read_csv(student_vle_file)
      vle_data_df = pd.read_csv(vle_file)
      assessments_df = pd.read_csv(assessments_file)
      student_assessment_df = pd.read_csv(student_assessment_file)
      return student_vle_data_df, vle_data_df, assessments_df, student_assessment_df

    except FileNotFoundError:

       print(f"Error: Input file '{input_csv}' not found.")

    except Exception as e:

       print(f"An error occurred: {e}")

def extract_interaction_features(dataframes):
    """Extracts interaction features from the interaction data.

    This function should be adapted to extract the relevant columns 
    from your interaction data.  The example provided uses the columns 
    'id_student', 'total_interactions', 'days_active', and 
    'forum_interactions'.
    """
    student_vle_data_df = dataframes['student_vle_file']
    vle_data_df = dataframes['vle_file']
    # Merge studentVle and vle dataframes
    merged_df = pd.merge(student_vle_data_df, vle_data_df, on='id_site')
    #interaction__features_df = merged_df[['id_student','total_interactions','days_active','forum_interactions' ]].copy()  # Create a copy to avoid SettingWithCopyWarning

    # 1. Effort and Persistence
    effort_persistence = merged_df.groupby('id_student').agg(total_interactions=('sum_click', 'sum'), days_active=('date', 'nunique'), date_std=('date', 'std')).fillna(0)
    
    # 2. Active Processing and Strategy Use
    

    learning_activities = merged_df[merged_df['activity_type'].str.contains('forum|quiz|ouwiki|glossary|dataplus|externalquiz|oucollaborate|questionnaire|htmlactivity|repeatactivity|sharedsubpage|ouelluminate|dualpane', case=False, na=False)].groupby('id_student')['sum_click'].sum().rename('learning_act').fillna(0)
    total_interactions_per_student = effort_persistence.total_interactions
    date_std_per_student = effort_persistence.date_std
    
    active_processing = pd.concat([learning_activities], axis=1).fillna(0)
    active_processing['learn_act_ratio'] = (active_processing['learning_act']) / total_interactions_per_student
    active_processing['learn_act_ratio'] = active_processing['learn_act_ratio'].replace([np.inf, -np.inf], 0).fillna(0)

   
    # 3. Monitoring and Self-Regulation
    def temporal_distribution(df):
        bins = np.linspace(df['date'].min(), df['date'].max(), 6)
        counts, _ = np.histogram(df['date'], bins=bins)
        return pd.Series(counts, index=['week1', 'week2', 'week3', 'week4', 'week5'])

    temporal_dist = merged_df.groupby('id_student').apply(temporal_distribution).fillna(0)
    temporal_dist['early_vs_late'] = (temporal_dist['week1'] + temporal_dist['week2']) / (temporal_dist['week4'] + temporal_dist['week5'] + 1e-10)
    temporal_dist = temporal_dist.replace([np.inf, -np.inf], 0)

    self_assessment_types = ['glossary', 'externalquiz', 'quiz', 'questionnaire', 'repeatactivity', 'dualpane']
    self_assessment_interactions = merged_df[merged_df['activity_type'].str.contains('|'.join(self_assessment_types), case=False, na=False)].groupby('id_student')['sum_click'].sum().fillna(0).replace([np.inf, -np.inf], 0).rename('self_ass_interact')
        

    self_assessment_ratio = (self_assessment_interactions / total_interactions_per_student).fillna(0).replace([np.inf, -np.inf], 0).rename('self_ass_ratio')
    temporal_dist = pd.concat([temporal_dist.early_vs_late, self_assessment_interactions, self_assessment_ratio], axis=1)
    
   
    # 4. Depth of Understanding and Critical Thinking (limited by data)
    
    depth_interactions = merged_df[merged_df['activity_type'].str.contains('forum|ouwiki|dataplus|oucollaborate|ouelluminate', case=False, na=False)].groupby('id_student')['sum_click'].sum().fillna(0).replace([np.inf, -np.inf], 0).rename('depth_interact')
    depth_interactions_ratio = (depth_interactions / total_interactions_per_student).fillna(0).replace([np.inf, -np.inf], 0).rename('depth_interact_ratio')
    depth_interactions = pd.concat([depth_interactions, depth_interactions_ratio], axis=1)
 
    # 5. Motivation and Interest
    early_engagement = merged_df[merged_df['date'] <= 14].groupby('id_student')['sum_click'].sum().fillna(0).replace([np.inf, -np.inf], 0).rename('early_engag')
    sustained_engagement = date_std_per_student.fillna(0).replace([np.inf, -np.inf], 0).rename('sustained_engag')
    
    motivation_interactions_types = ['resource', 'oucontent', 'url', 'forum', 'homepage', 'ouwiki', 'dataplus', 'oucollaborate', 'questionnaire', 'ouelluminate']
    motivation_interactions = merged_df[merged_df['activity_type'].str.contains('|'.join(motivation_interactions_types), case=False, na=False)].groupby('id_student')['sum_click'].sum().fillna(0).replace([np.inf, -np.inf], 0).rename('motiv_interact')
    
    motivation_interactions_ratio = (motivation_interactions / total_interactions_per_student).fillna(0).replace([np.inf, -np.inf], 0).rename('motiv_ratio')
    
    sustained_engagement = pd.concat([sustained_engagement, motivation_interactions, motivation_interactions_ratio], axis=1)


    all_interaction_features_df = pd.concat([effort_persistence, active_processing, temporal_dist, depth_interactions, early_engagement, sustained_engagement], axis=1).fillna(0)

    return all_interaction_features_df

def calculate_performance_features(dataframes):
    """Calculates performance features from assessment data.

    This function calculates the average assessment score for each student.
    You can expand this function to calculate other performance features,
    such as weighted averages, or scores on specific assessment types.
    """
    assessments_df = dataframes['assessments_file']
    student_assessment_df = dataframes['student_assessment_file']

    # Merge to get assessment scores and weights
    merged_df = pd.merge(student_assessment_df, assessments_df, on='id_assessment')

    # Calculate average assessment score
    avg_assessment_score = merged_df.groupby('id_student')['score'].mean().reset_index()
    avg_assessment_score.rename(columns={'score': 'avg_assessment_score'}, inplace=True)

    return avg_assessment_score


def merge_features(interaction_features_df, performance_features_df):
    """Merges interaction and performance features.

    This version of the function is simplified to only merge interaction and
    performance features.
    """
    combined_features_df = pd.merge(interaction_features_df, performance_features_df, on='id_student', how='left')
    return combined_features_df


def handle_missing_values(combined_features_df):
    """Handles missing values in the combined DataFrame."""
    # combined_features_df.fillna(combined_features_df.mean(), inplace=True)  #  Adapt as needed

    combined_features_df['avg_assessment_score'].fillna(0, inplace=True)
    return combined_features_df

# MAIN:
def main():
    """Main function to orchestrate feature extraction and processing."""

    # Define file paths.  Replace these with your actual file paths.
    file_paths = {
        'student_vle_file': 'dataset/anonymiseddata/studentVle.csv',  # Replace with your data file
        'vle_file': 'dataset/anonymiseddata/vle.csv',
        'assessments_file': 'dataset/anonymiseddata/assessments.csv',
        'student_assessment_file': 'dataset/anonymiseddata/studentAssessment.csv',
    }
    output_file = 'result_features/all_cognitive_engagement_features.csv'


    # 1. Load Data
    student_vle_data_df, vle_data_df, assessments_df, student_assessment_df = load_data(**file_paths)
    
    dataframes = {
        'student_vle_file': student_vle_data_df,
        'vle_file': vle_data_df,
        'assessments_file': assessments_df,
        'student_assessment_file': student_assessment_df,
    }

    # 2. Extract Features
    start_time = time.perf_counter()
    print(f"\nStarting Extracting Features ...")

    interaction_features_df = extract_interaction_features(dataframes)
    
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"\nExtracting Features Time: {execution_time:.2f} seconds")
        
    performance_features_df = calculate_performance_features(dataframes)
    
    # 3. Merge Features
    combined_features_df = merge_features(interaction_features_df, performance_features_df)


    # 4. Handle Missing Values
    combined_features_df = handle_missing_values(combined_features_df)

    # 5. Save extracted features to CSV
    combined_features_df.to_csv(output_file, index=False)
    print(f"Features extracted and saved to {output_file}")


if __name__ == "__main__":
    main()
