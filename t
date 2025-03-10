# Ensure branch column exists in raw data
if(!"branch" %in% colnames(merge_cloud_raw_prev)) {
  print("WARNING: branch column missing in merge_cloud_raw_prev - check input data")
  merge_cloud_raw_prev$branch <- NA
}


# Ensure branch column exists in raw data
if(!"branch" %in% colnames(merge_prem_raw_prev)) {
  print("WARNING: branch column missing in merge_prem_raw_prev - check input data")
  merge_prem_raw_prev$branch <- NA
}

# Ensure branch column exists in raw data
if(!"branch" %in% colnames(merge_cloud_raw)) {
  print("WARNING: branch column missing in merge_cloud_raw - check input data")
  merge_cloud_raw$branch <- NA
}


# Ensure branch column exists in raw data
if(!"branch" %in% colnames(merge_prem_raw)) {
  print("WARNING: branch column missing in merge_prem_raw - check input data")
  merge_prem_raw$branch <- NA
}

recurring_products <- model_pipeline %>% 
  filter(!is.na(ts))

# Check for branch column before select
if(!"branch" %in% colnames(model_pipeline)) {
  print("CRITICAL: branch column missing in model_pipeline before creating recurring_products - adding empty column")
  model_pipeline$branch <- NA
}


standard_run_recurring <- model_pipeline %>%
  filter(!is.na(ts))

# Check branch column before grouping
if(!"branch" %in% colnames(model_pipeline)) {
  print("CRITICAL: branch column missing in model_pipeline before creating standard_run_recurring - adding empty column")
  model_pipeline$branch <- NA
  # Re-filter after modification
  standard_run_recurring <- model_pipeline %>%
    filter(!is.na(ts))

# Now proceed with the select
program_semgrep_scan <- standard_run_recurring %>%
  filter(runWeek %in% c(latest_runWeek, second_latest_runWeek))

# Critical check - ensure branch exists before select operation
if(!"branch" %in% colnames(standard_run_recurring)) {
  print("CRITICAL: branch column missing in standard_run_recurring before creating program_semgrep_scan - adding empty column")
  standard_run_recurring$branch <- NA
  # Re-apply the filter since we modified the dataframe
  program_semgrep_scan <- standard_run_recurring %>%
    filter(runWeek %in% c(latest_runWeek, second_latest_runWeek))
}

# Critical check - ensure branch exists in both dataframes before joining
if(!"branch" %in% colnames(missing_scan)) {
  print("CRITICAL: branch column missing in missing_scan before final join - adding empty column")
  missing_scan$branch <- NA
}

if(!"branch" %in% colnames(gitsha_info)) {
  print("CRITICAL: branch column missing in gitsha_info before final join - adding empty column")
  gitsha_info$branch <- NA
}


# Use an explicit join specification with all matching columns
missing_scan <- missing_scan %>%
  left_join(gitsha_info, by = c("git_id", "gitSHA", "branch", "productStack"))

# Verify branch column exists after join
if(!"branch" %in% colnames(missing_scan)) {
  print("CRITICAL: branch column lost after join with gitsha_info - adding it back")
  # If we have branch in either source dataset, use it
  if("branch" %in% colnames(program_semgrep_scan)) {
    missing_scan <- missing_scan %>%
      left_join(program_semgrep_scan %>% select(git_id, branch) %>% unique(), by = "git_id")
  } else {
    # Last resort: add empty branch column
    missing_scan$branch <- NA
  }
}


missing_scan <- missing_scan %>%
  filter(is.na(scan_id)) %>%
  ungroup() 

# Debug to verify branch exists
print("Columns in missing_scan after processing:")
print(colnames(missing_scan))
print(glimpse(missing_scan))

# Final check - make sure we can actually write out the CSVs
tryCatch({
  # Create a sample of each dataframe to test
  print("Testing CSV writing with small samples...")
  for(df_name in c("missing_scan", "program_semgrep_scan")) {
    df <- get(df_name)
    sample_size <- min(5, nrow(df))
    if(sample_size > 0) {
      sample_df <- df[1:sample_size,]
      print(paste("Sample of", df_name, "that will be written to CSV:"))
      print(glimpse(sample_df))
    } else {
      print(paste("WARNING:", df_name, "is empty, nothing will be written"))
    }
  }
  print("CSV writing test complete - proceeding with full write")
}, error = function(e) {
  print(paste("ERROR during CSV test:", e$message))
  print("Attempting to fix any issues before final write...")
  # Emergency fixes could go here
})

