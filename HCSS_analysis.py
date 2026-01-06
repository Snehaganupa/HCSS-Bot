import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import os
import json
import glob

def load_data_from_file(filepath, year=None):
    """
    Load data from a specific Excel file and optionally add Year column.
    """
    data = pd.read_excel(filepath)
    # Ensure Date column is standardized
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        # Extract Year
        data["Year"] = data["Date"].dt.year
        data["Month"] = data["Date"].dt.month
        # Mark Rainy vs Non-Rainy (example: June–Sept = rainy)
        data["Season"] = data["Month"].apply(
            lambda m: "Rainy" if m in [6, 7, 8, 9] else "Non-Rainy"
        )
    else:
        raise ValueError("⚠️ Input file does not contain a 'Date' column")
    return data

def filter_data(data, exclude_cost_codes, exclude_job_codes):
    """
    Filter the data by excluding specified cost codes and job codes,
    and removing rows with zeros or NaNs in key columns.
    """
    data = data[~data['Cost Code'].isin(exclude_cost_codes) & ~data['Job Code'].isin(exclude_job_codes)]
    required_columns = ['Job Code', 'Cost Code', 'Unit', 'Actual Quantity']
    for col in required_columns:
        data = data[(data[col] != 0) & (data[col].notna())]
    return data

def clean_data(data, required_columns):
    """
    Remove rows with zeros or NaNs in specified columns.
    """
    for col in required_columns:
        data = data[(data[col] != 0) & (data[col].notna())]
    return data

def separate_data_by_row_count(data, groupby_cols, target_col):
    """
    Separate data into two sets: one with row count = 1 per group, one with row count > 1.
    """
    row_count = data.groupby(groupby_cols)[target_col].count().reset_index(name='count')
    row_count_1 = row_count[row_count['count'] == 1]
    row_count_more = row_count[row_count['count'] > 1]
    data_rc1 = pd.merge(row_count_1, data, on=groupby_cols, how='inner')
    data_rcm = pd.merge(row_count_more, data, on=groupby_cols, how='inner')
    return data_rc1, data_rcm

def calculate_productivity(data, quantity_col, labor_hours_col, productivity_col):
    data[productivity_col] = data.apply(
        lambda row: row[quantity_col] / row[labor_hours_col]
        if row[labor_hours_col] > 0 else np.nan, axis=1
    )
    return data

def calculate_unit_cost(data, cost_col, quantity_col, unit_cost_col):
    data[unit_cost_col] = data.apply(
        lambda row: row[cost_col] / row[quantity_col]
        if row[quantity_col] > 0 else np.nan, axis=1
    )
    return data

def calculate_statistics(data, groupby_cols, value_col, prefix):
    data = data[data[value_col].notna()]
    if data.empty:
        return pd.DataFrame(columns=groupby_cols + [
            f'Mean_of_{prefix}', f'Median_of_{prefix}',
            f'Mode_of_{prefix}', f'Std_of_{prefix}'
        ])
    grouped = data.groupby(groupby_cols)[value_col]
    mean = grouped.mean().reset_index(name=f'Mean_of_{prefix}')
    median = grouped.median().reset_index(name=f'Median_of_{prefix}')
    mode = grouped.apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).reset_index(name=f'Mode_of_{prefix}')
    std = grouped.std().reset_index(name=f'Std_of_{prefix}')
    stats = mean.merge(median, on=groupby_cols).merge(mode, on=groupby_cols).merge(std, on=groupby_cols)
    stats[f'Std_of_{prefix}'] = stats[f'Std_of_{prefix}'].fillna(0)
    return stats

def isolation_forest_analysis(data_actual, data_expected_std, groupby_cols, value_col, expected_std_col, prefix, contamination_range=(0.005, 0.05)):
    contamination = contamination_range[0]
    best_stats = None
    min_diff_sum = np.inf
    while contamination <= contamination_range[1]:
        data_copy = data_actual.copy()
        data_copy = data_copy[data_copy[value_col].notna()]
        if data_copy.empty:
            break
        iso = IsolationForest(contamination=contamination, random_state=42)
        data_copy['is_outlier'] = iso.fit_predict(data_copy[[value_col]])
        data_inliers = data_copy[data_copy['is_outlier'] == 1]
        if data_inliers.empty:
            contamination += 0.005
            continue
        stats = calculate_statistics(data_inliers, groupby_cols, value_col, prefix)
        stats['contamination'] = contamination
        stats = stats.merge(data_expected_std, on=groupby_cols, how='left')
        stats['difference'] = abs(stats[f'Std_of_{prefix}'] - stats[expected_std_col])
        total_diff = stats['difference'].sum()
        if total_diff < min_diff_sum:
            min_diff_sum = total_diff
            best_stats = stats.copy()
        contamination += 0.005
    if best_stats is not None:
        best_stats.drop(['difference', 'contamination'], axis=1, inplace=True)
        return best_stats
    else:
        return pd.DataFrame(columns=groupby_cols + [
            f'Mean_of_{prefix}', f'Median_of_{prefix}',
            f'Mode_of_{prefix}', f'Std_of_{prefix}'
        ])

def merge_results(unique_categories, actual_stats, expected_stats, groupby_cols):
    merged_data = unique_categories.merge(actual_stats, on=groupby_cols, how='outer')
    merged_data = merged_data.merge(expected_stats, on=groupby_cols, how='outer', suffixes=('', '_Expected'))
    merged_data.fillna(0, inplace=True)
    return merged_data

def main(groupby_cols, input_filename, output_filename=None, year=None):
    # Load the Excel file
    data = load_data_from_file(input_filename, year=year)
    # Parameters
    exclude_cost_codes = [
        '11-110', '11-123', '11-132', '11-135', '11-140', '11-150', '11-152',
        '11-153', '11-155', '11-156', '11-176', '11-182', '11-183', '11-184',
        '11-185', '11-186', '11-205', '11-206', '11-208', '11-210', '11-220',
        '11-222', '11-225', '11-230', '11-231', '11-232', '11-235', '11-236',
        '11-240', '11-245', '11-250', '11-300', '11-301', '11-388', '11-398',
        '11-399', '11-400', '11-401', '11-500', '11-900', '11-910', '11-920',
        '80-000', '80-001', '80-002', '80-003', '80-004', '80-005', '80-006',
        '80-007', '80-008', '80-009', '80-010', '80-011', '80-012', '80-013',
        '80-014', '80-015', '80-016', '80-017', '80-020', '80-021', '80-022',
        '80-023', '80-024', '80-025', '80-026', '80-027', '80-028', '80-029',
        '80-030', '80-031', '80-032', '80-035', '80-040', '80-050', '80-060',
        '80-075', '80-100', '80-101', '80-110', '80-111', '80-113', '80-114',
        '80-120', '80-140', '80-150', '80-200', '80-220', '80-221', '80-222',
        '80-250', '80-300', '81-000', '81-001', '81-002', '81-003', '81-004',
        '81-005', '81-006', '81-007', '81-008', '82-000', '82-001', '82-100',
        '82-105', '82-110', '82-116', '82-117', '82-118', '82-119', '82-121',
        '82-122', '84-001', '84-002', '85-001', '85-002', '85-003', '85-004',
        '85-005', '85-006', '85-007', '85-008', '85-009', '85-010', '85-021',
        '85-023', '85-100', '85-600', '85-700', '86-001', '86-004', '86-005',
        '86-010', '86-011', '86-012', '86-023', '86-034', '86-036', '86-103',
        '86-104', '86-105', '86-107', '86-109', '86-908', '87-001', '87-002',
        '87-003', '88-002', '88-003', '88-004', '88-005', '88-006', '88-007',
        '88-008', '88-009', '88-010', '88-011', '88-021', '88-100', '88-310',
        '88-500', '88-999', '89-001', '89-002', '89-003', '89-004', '89-005',
        '89-006', '89-007', '89-008', '89-009', '89-010', '89-011', '89-012',
        '89-013', '89-020', '89-021', '89-025', '89-030', '89-100', '89-110',
        '89-111', '89-112', '89-113', '89-114', '89-115', '89-116', '89-117',
        '89-135', '89-145', '89-150', '89-160', '89-172', '89-200', '89-201',
        '89-205', '89-206', '89-215', '89-216', '89-220', '89-225', '89-250',
        '89-255', '89-300', '89-301', '89-303', '89-304', '89-305', '89-307',
        '89-308', '89-310', '89-311', '89-312', '89-315', '89-350', '89-400',
        '89-404', '89-405', '89-406', '89-407', '89-408', '89-409', '89-410',
        '89-411', '89-412', '89-414', '89-416', '89-418', '89-420', '89-422',
        '89-450', '89-460', '89-465', '89-500', '89-550', '89-552', '89-556',
        '89-600', '89-700', '89-701', '89-702', '89-800', '90-001', '90-100',
        '99-000', '99-001', '99-002', '99-003', '99-004', '99-051', '99-100',
        '99-110', '99-400', '99-666', '99-700', '99-798', '99-799', '99-800',
        '99-801', '99-900', '99-990', '99-999'
    ]

    exclude_job_codes = ['CELERY', 'HCSSTEST', 'OFFICE', 'QC', 'SHOP', 'SURVEY', '(Blanks)']

    main_data = filter_data(data, exclude_cost_codes, exclude_job_codes)
    unique_categories = main_data[groupby_cols].drop_duplicates()

    # --- Productivity ---
    productivity_data = main_data[groupby_cols + ['Actual Quantity', 'Expected Labor Hours', 'Actual Labor Hours']]
    # Clean data
    productivity_data = clean_data(productivity_data, groupby_cols + ['Actual Quantity'])

    # Separate data by row count
    prod_actual_rc1, prod_actual_rcm = separate_data_by_row_count(productivity_data, groupby_cols, 'Actual Quantity')
    prod_expected_rc1 = prod_actual_rc1.copy()
    prod_expected_rcm = prod_actual_rcm.copy()

    # Calculate Actual and Expected Productivity
    prod_actual_rc1 = calculate_productivity(prod_actual_rc1, 'Actual Quantity', 'Actual Labor Hours', 'Actual Productivity')
    prod_expected_rc1 = calculate_productivity(prod_expected_rc1, 'Actual Quantity', 'Expected Labor Hours', 'Expected Productivity')
    prod_actual_rcm = calculate_productivity(prod_actual_rcm, 'Actual Quantity', 'Actual Labor Hours', 'Actual Productivity')
    prod_expected_rcm = calculate_productivity(prod_expected_rcm, 'Actual Quantity', 'Expected Labor Hours', 'Expected Productivity')

    # Remove zero or negative productivity
    prod_actual_rc1 = prod_actual_rc1[prod_actual_rc1['Actual Productivity'] > 0]
    prod_actual_rcm = prod_actual_rcm[prod_actual_rcm['Actual Productivity'] > 0]
    prod_expected_rcm = prod_expected_rcm[prod_expected_rcm['Expected Productivity'] > 0]

    # Expected Data Statistics
    expected_stats_rc1 = calculate_statistics(prod_expected_rc1, groupby_cols, 'Expected Productivity', 'Expected_Productivity')
    expected_stats_rcm = calculate_statistics(prod_expected_rcm, groupby_cols, 'Expected Productivity', 'Expected_Productivity')
    expected_stats = pd.concat([expected_stats_rc1, expected_stats_rcm], ignore_index=True)

    # Actual Data Statistics (Row count = 1)
    actual_stats_rc1 = calculate_statistics(prod_actual_rc1, groupby_cols, 'Actual Productivity', 'Actual_Productivity')

    # Actual Data Statistics (Row count > 1)
    expected_std = expected_stats[groupby_cols + ['Std_of_Expected_Productivity']]
    actual_stats_rcm = isolation_forest_analysis(
        prod_actual_rcm, expected_std, groupby_cols, 'Actual Productivity', 'Std_of_Expected_Productivity', 'Actual_Productivity'
    )

    # Combine Actual Data Statistics
    actual_stats = pd.concat([actual_stats_rc1, actual_stats_rcm], ignore_index=True)

    # Calculate Shift Productivity after outlier removal
    actual_stats['Mean_of_Actual_Shift_Productivity'] = actual_stats['Mean_of_Actual_Productivity'] * 9
    actual_stats['Median_of_Actual_Shift_Productivity'] = actual_stats['Median_of_Actual_Productivity'] * 9
    actual_stats['Mode_of_Actual_Shift_Productivity'] = actual_stats['Mode_of_Actual_Productivity'] * 9

    expected_stats['Mean_of_Expected_Shift_Productivity'] = expected_stats['Mean_of_Expected_Productivity'] * 9
    expected_stats['Median_of_Expected_Shift_Productivity'] = expected_stats['Median_of_Expected_Productivity'] * 9
    expected_stats['Mode_of_Expected_Shift_Productivity'] = expected_stats['Mode_of_Expected_Productivity'] * 9


    # --- Productivity ---
    # … productivity calculations …
    productivity_results = merge_results(unique_categories, actual_stats, expected_stats, groupby_cols)

    # -------------------------
    # Labor Cost Analysis
    # -------------------------
    labor_data = main_data[groupby_cols + ['Actual Quantity', 'Expected Labor Cost', 'Actual Labor Cost']]
    # Clean data
    labor_data = clean_data(labor_data, groupby_cols + ['Actual Quantity'])

    # Separate data by row count
    labor_actual_rc1, labor_actual_rcm = separate_data_by_row_count(labor_data, groupby_cols, 'Actual Quantity')
    labor_expected_rc1 = labor_actual_rc1.copy()
    labor_expected_rcm = labor_actual_rcm.copy()

    # Calculate Actual and Expected Unit Labor Cost
    labor_actual_rc1 = calculate_unit_cost(labor_actual_rc1, 'Actual Labor Cost', 'Actual Quantity',
                                           'Actual Unit Labor Cost')
    labor_expected_rc1 = calculate_unit_cost(labor_expected_rc1, 'Expected Labor Cost', 'Actual Quantity',
                                             'Expected Unit Labor Cost')
    labor_actual_rcm = calculate_unit_cost(labor_actual_rcm, 'Actual Labor Cost', 'Actual Quantity',
                                           'Actual Unit Labor Cost')
    labor_expected_rcm = calculate_unit_cost(labor_expected_rcm, 'Expected Labor Cost', 'Actual Quantity',
                                             'Expected Unit Labor Cost')

    # Remove zero or negative unit labor cost
    labor_actual_rc1 = labor_actual_rc1[labor_actual_rc1['Actual Unit Labor Cost'] > 0]
    labor_actual_rcm = labor_actual_rcm[labor_actual_rcm['Actual Unit Labor Cost'] > 0]
    labor_expected_rcm = labor_expected_rcm[labor_expected_rcm['Expected Unit Labor Cost'] > 0]

    # Expected Data Statistics
    expected_labor_stats_rc1 = calculate_statistics(labor_expected_rc1, groupby_cols, 'Expected Unit Labor Cost',
                                                    'Expected_Unit_Labor_Cost')
    expected_labor_stats_rcm = calculate_statistics(labor_expected_rcm, groupby_cols, 'Expected Unit Labor Cost',
                                                    'Expected_Unit_Labor_Cost')
    expected_labor_stats = pd.concat([expected_labor_stats_rc1, expected_labor_stats_rcm], ignore_index=True)

    # Actual Data Statistics (Row count = 1)
    actual_labor_stats_rc1 = calculate_statistics(labor_actual_rc1, groupby_cols, 'Actual Unit Labor Cost',
                                                  'Actual_Unit_Labor_Cost')

    # Actual Data Statistics (Row count > 1)
    expected_labor_std = expected_labor_stats[groupby_cols + ['Std_of_Expected_Unit_Labor_Cost']]
    actual_labor_stats_rcm = isolation_forest_analysis(
        labor_actual_rcm, expected_labor_std, groupby_cols, 'Actual Unit Labor Cost', 'Std_of_Expected_Unit_Labor_Cost',
        'Actual_Unit_Labor_Cost'
    )

    # Combine Actual Data Statistics
    actual_labor_stats = pd.concat([actual_labor_stats_rc1, actual_labor_stats_rcm], ignore_index=True)

    # Merge Results
    labor_results = merge_results(unique_categories, actual_labor_stats, expected_labor_stats, groupby_cols)

    # -------------------------
    # Equipment Cost Analysis
    # -------------------------
    equipment_data = main_data[groupby_cols + ['Actual Quantity', 'Expected Equipment Cost', 'Actual Equipment Cost']]
    # Clean data
    equipment_data = clean_data(equipment_data, groupby_cols + ['Actual Quantity'])

    # Separate data by row count
    equip_actual_rc1, equip_actual_rcm = separate_data_by_row_count(equipment_data, groupby_cols, 'Actual Quantity')
    equip_expected_rc1 = equip_actual_rc1.copy()
    equip_expected_rcm = equip_actual_rcm.copy()

    # Calculate Actual and Expected Unit Equipment Cost
    equip_actual_rc1 = calculate_unit_cost(equip_actual_rc1, 'Actual Equipment Cost', 'Actual Quantity',
                                           'Actual Unit Equipment Cost')
    equip_expected_rc1 = calculate_unit_cost(equip_expected_rc1, 'Expected Equipment Cost', 'Actual Quantity',
                                             'Expected Unit Equipment Cost')
    equip_actual_rcm = calculate_unit_cost(equip_actual_rcm, 'Actual Equipment Cost', 'Actual Quantity',
                                           'Actual Unit Equipment Cost')
    equip_expected_rcm = calculate_unit_cost(equip_expected_rcm, 'Expected Equipment Cost', 'Actual Quantity',
                                             'Expected Unit Equipment Cost')

    # Remove zero or negative unit equipment cost
    equip_actual_rc1 = equip_actual_rc1[equip_actual_rc1['Actual Unit Equipment Cost'] > 0]
    equip_actual_rcm = equip_actual_rcm[equip_actual_rcm['Actual Unit Equipment Cost'] > 0]
    equip_expected_rcm = equip_expected_rcm[equip_expected_rcm['Expected Unit Equipment Cost'] > 0]

    # Expected Data Statistics
    expected_equip_stats_rc1 = calculate_statistics(equip_expected_rc1, groupby_cols, 'Expected Unit Equipment Cost',
                                                    'Expected_Unit_Equipment_Cost')
    expected_equip_stats_rcm = calculate_statistics(equip_expected_rcm, groupby_cols, 'Expected Unit Equipment Cost',
                                                    'Expected_Unit_Equipment_Cost')
    expected_equip_stats = pd.concat([expected_equip_stats_rc1, expected_equip_stats_rcm], ignore_index=True)

    # Actual Data Statistics (Row count = 1)
    actual_equip_stats_rc1 = calculate_statistics(equip_actual_rc1, groupby_cols, 'Actual Unit Equipment Cost',
                                                  'Actual_Unit_Equipment_Cost')

    # Actual Data Statistics (Row count > 1)
    expected_equip_std = expected_equip_stats[groupby_cols + ['Std_of_Expected_Unit_Equipment_Cost']]
    actual_equip_stats_rcm = isolation_forest_analysis(
        equip_actual_rcm, expected_equip_std, groupby_cols, 'Actual Unit Equipment Cost',
        'Std_of_Expected_Unit_Equipment_Cost', 'Actual_Unit_Equipment_Cost'
    )

    # Combine Actual Data Statistics
    actual_equip_stats = pd.concat([actual_equip_stats_rc1, actual_equip_stats_rcm], ignore_index=True)

    # Merge Results
    equipment_results = merge_results(unique_categories, actual_equip_stats, expected_equip_stats, groupby_cols)

    # -------------------------
    # Final Merge
    # -------------------------
    final_data = productivity_results.merge(labor_results, on=groupby_cols, how='outer', suffixes=('', '_Labor'))
    final_data = final_data.merge(equipment_results, on=groupby_cols, how='outer', suffixes=('', '_Equipment'))

    # Fill NaN values with zero
    final_data.fillna(0, inplace=True)

    # Drop standard deviation columns before exporting
    std_columns = [col for col in final_data.columns if 'Std_of_' in col]
    final_data.drop(columns=std_columns, inplace=True)

    # Round float columns
    float_cols = final_data.select_dtypes(include=['float64', 'float32']).columns
    final_data[float_cols] = final_data[float_cols].round(4)

    # Standardize column names
    final_data.columns = final_data.columns.str.replace(' ', '_', regex=False)
    final_data.columns = final_data.columns.str.replace('-', '_', regex=False)
    final_data.columns = final_data.columns.str.lower()

    # Optionally save to Excel and JSON files
    # Don't write here; let process_all_years handle Excel
    return final_data

def assign_season(date):
    """Return 'Rainy' or 'Non-Rainy' based on month."""
    if pd.isna(date):
        return "Unknown"
    month = pd.to_datetime(date).month
    # Example rule — adjust if your rainy months differ
    if month in [6, 7, 8, 9]:
        return "Rainy"
    else:
        return "Non-Rainy"

def process_all_years(file_list, output_prefix):
    # -----------------------------
    # Step 1: Combine all yearly files
    # -----------------------------
    combined_data = []
    for file in file_list:
        print(f"📂 Loading {file}")
        year_part = file.split("_")[-1].replace(".xlsx", "")
        try:
            year = int(year_part[-2:]) + 2000
        except ValueError:
            year = None
        data = pd.read_excel(file)
        if year:
            data["Year"] = year

        # ✅ Add Season column based on Date
        if "Date" in data.columns:
            data["Season"] = data["Date"].apply(assign_season)
        else:
            data["Season"] = "Unknown"
        combined_data.append(data)

    combined_df = pd.concat(combined_data, ignore_index=True)
    print(f"✅ Combined {len(file_list)} files, total rows: {len(combined_df)}")

    # Save temporary combined file (so main() can read it normally)
    temp_combined_file = "HCSS_combined_temp.xlsx"
    combined_df.to_excel(temp_combined_file, index=False)

    # -----------------------------
    # Step 2: Run both analyses
    # -----------------------------
    cc_groupby = ['Cost Code', 'Unit', 'Cost Code Description', 'Season']
    jc_groupby = ['Job Code', 'Cost Code', 'Unit', 'Cost Code Description', 'Season']

    print("📊 Running cost code + season level analysis (all years combined)...")
    cc_result = main(cc_groupby, temp_combined_file, f"{output_prefix}_cost_code_season")

    print("📊 Running job code + season level analysis (all years combined)...")
    jc_result = main(jc_groupby, temp_combined_file, f"{output_prefix}_job_code_season")

    # -----------------------------
    # Step 3: Save outputs
    # -----------------------------
    if isinstance(cc_result, pd.DataFrame) and not cc_result.empty:
        cc_result.to_excel(f"{output_prefix}_cost_code_season.xlsx", index=False)
        print(f"✅ Saved → {output_prefix}_cost_code_season.xlsx")

    if isinstance(jc_result, pd.DataFrame) and not jc_result.empty:
        jc_result.to_excel(f"{output_prefix}_job_code_season.xlsx", index=False)
        print(f"✅ Saved → {output_prefix}_job_code_season.xlsx")

    print("🎯 All-years combined analysis completed successfully.")




if __name__ == "__main__":
    file_list = [
        "CostCodeDetailReport_13.xlsx",
        "CostCodeDetailReport_14.xlsx",
        "CostCodeDetailReport_15.xlsx",
        "CostCodeDetailReport_16.xlsx",
        "CostCodeDetailReport_17.xlsx",
        "CostCodeDetailReport_18.xlsx",
        "CostCodeDetailReport_19.xlsx",
        "CostCodeDetailReport_20.xlsx",
        "CostCodeDetailReport_21-2.xlsx",
        "CostCodeDetailReport_22-2.xlsx",
        "CostCodeDetailReport_23-2.xlsx",
        "CostCodeDetailReport_24-2.xlsx",
        "CostCodeDetailReport_25-2.xlsx"
    ]
    process_all_years(file_list, "main_data_analyzed")

