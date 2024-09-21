import pandas as pd

def convert_data(old_file='simulation_data_old.csv', new_file='simulation_data.csv'):
    # Read the old data
    df_old = pd.read_csv(old_file)
    # Compute N_prime from lambda_prime and S_prime
    df_old['N_prime'] = (df_old['lambda_prime'] * df_old['S_prime']).round().astype(int)
    # Select only the required columns
    df_new = df_old[['S_prime', 'N_prime', 'iteration']]
    # Save to new file
    df_new.to_csv(new_file, index=False)
    print(f"Converted data saved to {new_file}")

if __name__ == "__main__":
    convert_data()
