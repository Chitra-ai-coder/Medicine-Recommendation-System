import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast

def visualize_kaggle_data():
    print("📊 Loading Kaggle Dataset and generating graphs...")

    try:
        train_df = pd.read_csv('archive/Training.csv')
        meds_df = pd.read_csv('archive/medications.csv')
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    sns.set_theme(style="whitegrid")

    # ---------------------------------------------------------
    # Plot 1: Top 20 Most Common Symptoms
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 8))
    symptoms_only = train_df.drop(columns=['prognosis'])
    symptoms_only = symptoms_only.loc[:, ~symptoms_only.columns.str.contains('^Unnamed')]
    
    top_symptoms = symptoms_only.sum().sort_values(ascending=False).head(20)
    top_symptoms.index = top_symptoms.index.str.replace('_', ' ').str.title()

    sns.barplot(x=top_symptoms.values, y=top_symptoms.index, hue=top_symptoms.index, palette='viridis', legend=False)
    plt.title('Top 20 Most Common Symptoms', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('graph1_top_symptoms.png') # <--- SAVING INSTEAD OF SHOWING
    plt.close()

    # ---------------------------------------------------------
    # Plot 2: Distribution of Diseases
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 10))
    disease_counts = train_df['prognosis'].value_counts()
    
    sns.barplot(x=disease_counts.values, y=disease_counts.index, hue=disease_counts.index, palette='magma', legend=False)
    plt.title('Distribution of Diseases', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('graph2_diseases.png') # <--- SAVING INSTEAD OF SHOWING
    plt.close()

    # ---------------------------------------------------------
    # Plot 3: Number of Medications Prescribed per Disease
    # ---------------------------------------------------------
    plt.figure(figsize=(14, 8))
    
    def count_meds(med_string):
        try:
            return len(ast.literal_eval(med_string))
        except:
            return 0

    meds_df['Num_Medications'] = meds_df['Medication'].apply(count_meds)
    meds_df_sorted = meds_df.sort_values(by='Num_Medications', ascending=False)

    sns.barplot(x='Disease', y='Num_Medications', data=meds_df_sorted, hue='Disease', palette='coolwarm', legend=False)
    plt.title('Number of Medications Prescribed per Disease', fontsize=16, fontweight='bold')
    plt.xticks(rotation=90) 
    plt.tight_layout()
    plt.savefig('graph3_medications.png') # <--- SAVING INSTEAD OF SHOWING
    plt.close()

    print("✅ All 3 graphs have been saved as PNG files in your ml_training folder!")

if __name__ == "__main__":
    visualize_kaggle_data()