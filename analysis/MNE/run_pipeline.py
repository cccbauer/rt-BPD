"""
Quick Start Script
==================
"""

import os
import glob
from multi_subject_pipeline import MultiSubjectPDAPipeline

# Create subject list
subject_runs = []
edf_files = sorted(glob.glob('../edfs/sub-*_task-feedback-run*.edf'))

for edf_file in edf_files:
    basename = os.path.basename(edf_file)
    parts = basename.split('_')
    subject_id = parts[0]
    run_num = basename.split('-run')[1].replace('.edf', '')
    
    task_file = f'./task_output/{subject_id}_DMN_Feedback_run{run_num}_roi_outputs.csv'
    
    if os.path.exists(task_file):
        subject_run_id = f'{subject_id}_run{run_num}'
        subject_runs.append((subject_run_id, edf_file, task_file))
        print(f"Found: {subject_run_id}")

print(f"\nTotal runs found: {len(subject_runs)}")

# Split into train/test
subjects = {}
for subject_run_id, edf, task in subject_runs:
    subject = subject_run_id.split('_run')[0]
    if subject not in subjects:
        subjects[subject] = []
    subjects[subject].append((subject_run_id, edf, task))

subject_list = list(subjects.keys())
n_train = int(0.8 * len(subject_list))

train_subjects = subject_list[:n_train]
test_subjects = subject_list[n_train:]

train_runs = []
test_runs = []

for subject in train_subjects:
    train_runs.extend(subjects[subject])

for subject in test_subjects:
    test_runs.extend(subjects[subject])

print(f"Training: {len(train_subjects)} subjects, {len(train_runs)} runs")
print(f"Testing: {len(test_subjects)} subjects, {len(test_runs)} runs")

# Run pipeline
pipeline = MultiSubjectPDAPipeline(
    base_dir='../',
    output_dir='./results_quickstart'
)

print("\nProcessing training subjects...")
pipeline.process_all_subjects(train_runs)

print("\nTraining models...")
pipeline.combine_subjects_data()
results = pipeline.train_models()

pipeline.save_model('quickstart_model.pkl')

print("\nPipeline complete!")
