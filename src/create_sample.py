import pandas as pd
df = pd.read_csv('global_training_set.csv')

# Sample 500 rows randomly to create a smaller dataset
sampled_df = df.sample(n=500, random_state=42)


# Save the sampled dataset
sampled_file_path = "sampled_training_set.csv"
sampled_df.to_csv(sampled_file_path, index=False)
