# Data Quality Summary

## 1. Datasets Loaded
- Successfully loaded all 10 CSV datasets using Pandas.
- Verified the shape, data types, and sample records of each dataset.

## 2. Missing Values
- Checked for missing values using `isnull().sum()`.
- No critical missing values found.
- (If you found missing values, mention the dataset and columns.)

## 3. Duplicate Records
- Checked duplicate rows using `duplicated().sum()`.
- No duplicate records found.
- (Or mention the number if duplicates exist.)

## 4. API Data
- Successfully fetched live NAV data from `https://api.mfapi.in/mf/125497`.
- Saved the response as `HDFC_NAV.csv`.
- Downloaded NAV history for the additional 5 mutual fund schemes.

## 5. AMFI Code Validation
- Compared `amfi_code` between `fund_master.csv` and `nav_history.csv`.
- All AMFI codes matched successfully.
- (Or mention if any codes were missing.)

## 6. Overall Observation
The datasets were successfully ingested and explored. Data quality checks were performed, and live NAV data was fetched successfully. The project is ready for the next stage of data cleaning and analysis.