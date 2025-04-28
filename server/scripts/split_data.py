# split_data.py
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

def main():
    input_path = sys.argv[1]
    train_path = sys.argv[2]
    val_path = sys.argv[3]
    test_size = float(sys.argv[5]) if '--test-size' in sys.argv else 0.2
    
    df = pd.read_csv(input_path)
    train, val = train_test_split(df, test_size=test_size)
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)

if __name__ == "__main__":
    main()
