# Script generado desde notebook

def main():
    import pandas as pd
    files = [
        r"C:\Users\alvar\Documents\Projects\Hydrapig_Analysis\data\Processed\clustering\clustering_week_1.csv",
        r"C:\Users\alvar\Documents\Projects\Hydrapig_Analysis\data\Processed\clustering\clustering_week_2.csv",
        r"C:\Users\alvar\Documents\Projects\Hydrapig_Analysis\data\Processed\clustering\clustering_week_3.csv",
        r"C:\Users\alvar\Documents\Projects\Hydrapig_Analysis\data\Processed\clustering\clustering_week_4.csv",
        r"C:\Users\alvar\Documents\Projects\Hydrapig_Analysis\data\Processed\clustering\clustering_week_5.csv",
        r"C:\Users\alvar\Documents\Projects\Hydrapig_Analysis\data\Processed\clustering\clustering_week_6.csv"
    ]
    weekly_data = []
    for i, file in enumerate(files, start=1):
        df = pd.read_csv(file)
        df['Week'] = i  # Add the week number
        weekly_data.append(df)
    clustering_data = pd.concat(weekly_data, ignore_index=True)
    clustering_data.to_csv(r"C:\Users\alvar\Documents\Projects\Hydrapig_Analysis\data\Processed\clustering\clustering_data.csv", index=False)
if __name__ == "__main__":
    main()
