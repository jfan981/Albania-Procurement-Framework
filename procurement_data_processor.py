import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURATION ---

# Define the Albanian-to-English header map
HEADER_MAP = {
    'Autoriteti_kontraktues': 'contracting_authority',
    'Numri_i_references': 'reference_number',
    'Objekti_i_prokurimit': 'procurement_object',
    'Lloji_i_procedures': 'procedure_type',
    'Tipi_i_kontrates': 'contract_type',
    'Lloji_i_marreveshjes_kuader': 'framework_agreement_type',
    'Fondi_limit': 'limit_fund',
    'Data_e_publikimit': 'publication_date',
    'Data_e_hapjes': 'opening_date',
    'Data_e_mbylljes': 'closing_date',
    'Anulluar': 'is_canceled',
    'Arsyeja_e_anullimit': 'cancellation_reason',
    'Pezulluar': 'is_suspended',
    'Fituesi': 'winner_name',
    'NIPT_i_fituesit': 'winner_nipt',
    'Vlera_e_fituesit': 'winner_value',
    'Vlera_e_fituesit_ne_lidhjen_e_kontrates': 'winner_value_at_contract_signing',
    'Lidhja_e_kontrates_me_TVSH': 'contract_signed_with_vat',
    'Numri_i_ofertave_te_dorezuara': 'num_bids_submitted',
    'Numri_i_ofertave_te_kualifikuara': 'num_bids_qualified',
    'Kodet_CPV': 'cpv_codes'
}

# Source file - 2025
SOURCE_FILES = [
    'Procedurat_e_prokurimit_-_viti_2025_-_Gjeneruar_me_02112025.csv'
]

# --- 2. DATA LOADING AND TRANSLATION ---

def load_and_combine_data(files):
    """Loads, translates headers, and combines all CSV files."""
    all_dataframes = []
    for f in files:
        if not os.path.exists(f):
            print(f"Warning: File not found - {f}. Skipping.")
            continue
        try:
            # Load CSV
            df = pd.read_csv(f)

            # Rename columns based on the map
            df = df.rename(columns=HEADER_MAP)

            all_dataframes.append(df)
            print(f"Successfully loaded and translated {f}")

        except Exception as e:
            print(f"Error processing file {f}: {e}")

    if not all_dataframes:
        print("No data loaded. Exiting.")
        return None

    # Combine all data into a single DataFrame
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    return combined_df

# --- 3. DATA CLEANING ---

def clean_data(df):
    """Cleans data types for key columns."""
    print("Cleaning data...")

    # Clean currency fields: remove quotes, commas, and convert to numeric
    currency_cols = ['limit_fund', 'winner_value', 'winner_value_at_contract_signing']
    for col in currency_cols:
        if col in df.columns:
            # Ensure the column is a string before trying string operations
            df[col] = df[col].astype(str).str.replace('"', '').str.replace(',', '')
            # Coerce errors will turn unparseable strings into NaT (for dates) or NaN (for numbers)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Clean simple numeric fields
    numeric_cols = ['num_bids_submitted', 'num_bids_qualified']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Fill NaNs in these specific columns with 0 before calculations
            df[col] = df[col].fillna(0)

    # Convert date fields to datetime objects
    date_cols = ['publication_date', 'opening_date', 'closing_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%d.%m.%Y', errors='coerce')

    # Fill NaNs in key fields for calculations
    df['winner_value'] = df['winner_value'].fillna(0)
    df['winner_nipt'] = df['winner_nipt'].fillna('UNKNOWN')

    # Map 'is_canceled' to boolean for easier calculations
    df['is_canceled_bool'] = df['is_canceled'].map({'Po': True, 'Jo': False}).fillna(False)

    print("Data cleaning complete.")
    return df

# --- 4. FEATURE ENGINEERING ---

def create_features(df):
    """Creates new features as requested."""
    print("Engineering new features...")

    # === Tender-Level Features ===
    print("  - Creating tender-level features...")

    # 1. fund_usage
    df['fund_usage'] = df['winner_value'] / df['limit_fund']
    # Handle division by zero (resulting in inf)
    df['fund_usage'] = df['fund_usage'].replace([np.inf, -np.inf], np.nan)
    # --- Cap fund_usage at 1.0 ---
    df['fund_usage'] = df['fund_usage'].clip(upper=1.0)

    # 2. value_changed_at_contract_signing
    # Automatically be NaN if 'winner_value_at_contract_signing' is NaN
    df['value_changed_at_contract_signing'] = df['winner_value_at_contract_signing'] - df['winner_value']

    # 3. is_single_bidder
    df['is_single_bidder'] = df['num_bids_submitted'] == 1

    # 4. is_single_qualified_bid
    df['is_single_qualified_bid'] = df['num_bids_qualified'] == 1

    # 5. tender_duration_days
    df['tender_duration_days'] = (df['closing_date'] - df['publication_date']).dt.days

    # 6. is_over_budget
    df['is_over_budget'] = df['winner_value'] > df['limit_fund']

    # === Company-Wide Aggregate Features ===
    print("  - Creating company-wide aggregate features...")

    # Create a dataframe of just the "wins" to aggregate
    wins_df = df[df['winner_nipt'] != 'UNKNOWN'].copy()

    if not wins_df.empty:
        # Group by company NIPT and aggregate statistics
        company_stats = wins_df.groupby('winner_nipt').agg(
            company_total_wins=('winner_nipt', 'count'),
            company_total_value=('winner_value', 'sum'),
            company_avg_fund_usage=('fund_usage', 'mean'), # mean ignores NaNs
            company_total_single_bid_wins=('is_single_bidder', 'sum'),
            company_total_canceled_wins=('is_canceled_bool', 'sum')
        )

        # Calculate rate-based features
        company_stats['company_single_bid_win_rate'] = company_stats['company_total_single_bid_wins'] / company_stats['company_total_wins']
        company_stats['company_cancellation_rate'] = company_stats['company_total_canceled_wins'] / company_stats['company_total_wins']

        # Rename columns to avoid conflicts and make merging clear
        company_stats.columns = [
            'company_total_wins', 'company_total_value', 'company_avg_fund_usage',
            'company_total_single_bid_wins', 'company_total_canceled_wins',
            'company_single_bid_win_rate', 'company_cancellation_rate'
        ]

        # Merge these new company stats back into the main dataframe
        df = df.merge(company_stats, on='winner_nipt', how='left')

        # Fill NaNs for rows with no company (e.g., 'UNKNOWN' NIPT) or for companies with no wins
        company_stat_cols = company_stats.columns
        df[company_stat_cols] = df[company_stat_cols].fillna(0)

    else:
        print("  - No wins found in dataset, skipping company-wide features.")
        # Create empty columns if no wins exist
        empty_cols = [
            'company_total_wins', 'company_total_value', 'company_avg_fund_usage',
            'company_total_single_bid_wins', 'company_total_canceled_wins',
            'company_single_bid_win_rate', 'company_cancellation_rate'
        ]
        for col in empty_cols:
            df[col] = 0

    # Clean up helper columns
    df = df.drop(columns=['is_canceled_bool'])

    print("Feature engineering complete.")
    return df

# --- 5. DATA SPLITTING AND SAVING ---

def split_and_save_data(df):
    """Splits data into 80/10/10 train/validation/test sets and saves to CSV."""
    print("Splitting data into train, validation, and test sets...")

    # First split: 80% train, 20% temp (for val/test)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

    # Second split: Split the 20% temp into 50% validation and 50% test
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"\nTotal rows: {len(df)}")
    print(f"Train set: {len(train_df)} rows (~{len(train_df)/len(df):.0%})")
    print(f"Validation set: {len(val_df)} rows (~{len(val_df)/len(df):.0%})")
    print(f"Test set: {len(test_df)} rows (~{len(test_df)/len(df):.0%})")

    # Save the files
    try:
        print("\nSaving files...")
        # Save the full engineered dataset
        df.to_csv('all_data_engineered.csv', index=False)
        print("Successfully saved all_data_engineered.csv")

        # Save the splits
        train_df.to_csv('train_data.csv', index=False)
        print("Successfully saved train_data.csv")

        val_df.to_csv('validation_data.csv', index=False)
        print("Successfully saved validation_data.csv")

        test_df.to_csv('test_data.csv', index=False)
        print("Successfully saved test_data.csv")

        print("\nAll tasks complete.")

    except Exception as e:
        print(f"Error saving files: {e}")

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    # Step 1: Load and combine
    combined_data = load_and_combine_data(SOURCE_FILES)

    if combined_data is not None:
        # Step 2: Clean data
        cleaned_data = clean_data(combined_data)

        # Step 3: Create features
        engineered_data = create_features(cleaned_data)

        # Step 4: Split and save
        split_and_save_data(engineered_data)