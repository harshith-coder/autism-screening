import pandas as pd
import numpy as np

# Create realistic sample autism screening dataset
np.random.seed(42)
n_samples = 704

# Features based on typical autism screening questionnaires
data = {
    'A1_prefer_detail_not_big_picture': np.random.randint(0, 2, n_samples),
    'A2_must_have_sameness': np.random.randint(0, 2, n_samples),
    'A3_prefer_reading_systematically': np.random.randint(0, 2, n_samples),
    'A4_feel_anxious_in_social': np.random.randint(0, 2, n_samples),
    'A5_prefer_talking_one_to_one': np.random.randint(0, 2, n_samples),
    'A6_notice_small_changes': np.random.randint(0, 2, n_samples),
    'A7_trouble_focus_on_changing': np.random.randint(0, 2, n_samples),
    'A8_often_daydream': np.random.randint(0, 2, n_samples),
    'A9_focused_on_one_topic': np.random.randint(0, 2, n_samples),
    'A10_difficult_small_talk': np.random.randint(0, 2, n_samples),
    'age': np.random.randint(18, 80, n_samples),
    'gender': np.random.choice(['M', 'F'], n_samples),
    'ethnicity': np.random.choice(['White', 'Asian', 'Black', 'Others'], n_samples),
    'jundice': np.random.choice(['yes', 'no'], n_samples),
    'autism_family_member': np.random.choice(['yes', 'no'], n_samples),
    'country': np.random.choice(['USA', 'UK', 'Canada', 'India'], n_samples),
    'used_app_before': np.random.choice(['yes', 'no'], n_samples),
    'screening_type': np.random.choice(['Questionnaire', 'Interview'], n_samples),
}

autism_score = (data['A1_prefer_detail_not_big_picture'] + 
                data['A2_must_have_sameness'] + 
                data['A4_feel_anxious_in_social'] +
                data['A9_focused_on_one_topic'] +
                data['A10_difficult_small_talk'])

class_binary = (autism_score >= 3).astype(int)
data['Class'] = ['YES' if x == 1 else 'NO' for x in class_binary]

df = pd.DataFrame(data)
df.to_csv('data/autism_screening.csv', index=False)
print(f'✅ Sample dataset created!')
print(f'   Records: {len(df)}')
print(f'   Features: {len(df.columns)}')
print(f'   Saved to: data/autism_screening.csv')
print(f'\nClass Distribution:')
print(df['Class'].value_counts())
