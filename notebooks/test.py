import pandas as pd

df = pd.read_csv('../results/experiment_results.csv')
print(df.columns)
df_sorted = df.sort_values(by='f1_score', ascending=False)

# 2. Extract metrics from the top row for columns 'c' and 'd'
top_row = df_sorted.iloc[0]
f1_score = top_row['f1_score']
accuracy = top_row['accuracy']
precision = top_row['precision']
recall = top_row['recall']
print(f"F1 Score: {f1_score}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")