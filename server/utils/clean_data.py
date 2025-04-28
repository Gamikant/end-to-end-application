# clean_data.py
import sys
import pandas as pd
from sklearn.impute import KNNImputer
import os

def main(input_path, output_path, file_ext=None):
    try:
        # Determine file extension
        ext = file_ext.lower() if file_ext else os.path.splitext(input_path)[1].lower()
        if not ext:
            raise ValueError("No file extension provided")

        # Read file based on extension
        if ext == '.csv':
            df = pd.read_csv(input_path)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(input_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        # Identify numeric columns using pandas
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # KNN Imputation
        imputer = KNNImputer(n_neighbors=3)
        df_imputed = df.copy()

        if numeric_cols:
            df_imputed[numeric_cols] = pd.DataFrame(
                imputer.fit_transform(df[numeric_cols]),
                columns=numeric_cols
            )

        # Save cleaned data (non-numeric columns remain untouched)
        df_imputed.to_csv(output_path, index=False)
        print("CLEANING_SUCCESS")

    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python clean_data.py <input_path> <output_path> [file_extension]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
