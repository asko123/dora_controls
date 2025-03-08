 Save branch column specifically
branch_data <- NULL
if("branch" %in% model_pipeline_cols) {
  print("Saving branch data before join")
  branch_data <- model_pipeline %>% 
    select(git_id, branch) %>% 
    unique()
  
  # Show a sample of the branch data to confirm it's valid
  print("Sample of branch data being saved:")
  print(head(branch_data))
} else {
  print("WARNING: branch column not found in model_pipeline before join")
}

# Perform the join with a different approach to preserve columns
# Instead of doing a left_join(model_pipeline), let's try a more explicit approach
model_pipeline_new <- model_spr_git %>% 
  select(git_id, repository.repoGuid, stack.name, current_scope, creationDate) %>% 
  unique()

print("Columns in model_pipeline_new before merge:")
print(colnames(model_pipeline_new))

# Now join with model_pipeline in a way that preserves columns
print("Merging model_pipeline_new with model_pipeline...")
model_pipeline_merged <- model_pipeline_new %>%
  left_join(model_pipeline, by = "git_id")

print("Columns after merge:")
print(colnames(model_pipeline_merged))

# Finish the transformation
model_pipeline <- model_pipeline_merged %>%
  select(-stack.name) %>% 
  unique()

# Check if branch column was lost in the join
if(!"branch" %in% colnames(model_pipeline) && !is.null(branch_data)) {
  print("WARNING: branch column was lost in the join, adding it back")
  # Add it back
  model_pipeline <- model_pipeline %>%
    left_join(branch_data, by = "git_id")
  
  # Verify branch was added
  print("Columns after restoring branch:")
  print(colnames(model_pipeline))
} else if(!"branch" %in% colnames(model_pipeline)) {
  print("CRITICAL WARNING: branch column wasn't present or couldn't be restored, creating empty one")
  model_pipeline$branch <- NA
  print("Added empty branch column")
}

# Debug model_pipeline after join
print("Columns in model_pipeline after join with model_spr_git (with branch fix):")
print(colnames(model_pipeline))

# ============================ ESTABLISH THE RUN WEEK FOR ALL THE PRODUCTS IN SCOPE ============================

# Double check that branch exists
if(!"branch" %in% colnames(model_pipeline)) {
  print("CRITICAL ERROR: Branch column is still missing after fixes. Proceeding with empty branch column.")
  model_pipeline$branch <- NA
}

# CRITICAL: Verify we have the important columns before continuing
required_columns <- c("git_id", "branch", "gitSHA", "productStack", "ts")
missing_columns <- required_columns[!required_columns %in% colnames(model_pipeline)]

if(length(missing_columns) > 0) {
  print(paste("CRITICAL: Missing required columns:", paste(missing_columns, collapse=", ")))
  print("Current columns in model_pipeline:")
  print(colnames(model_pipeline))
  
  # For each missing column, add a placeholder
  for(col in missing_columns) {
    print(paste("Adding empty column for:", col))
    model_pipeline[[col]] <- NA
  }
}

# Verify all required columns now exist
print("Columns after ensuring required columns exist:")
print(colnames(model_pipeline))

# We split into 2 dataframes: one for products that have history and one that never have previous records
recurring_products <- model_pipeline %>% 
  filter(!is.na(ts)) %>%
  select(git_id, gitSHA, productStack, branch) %>%
  unique()

# Debug recurring_products to see what columns it has
print("Columns in recurring_products:")
print(colnames(recurring_products))

# For recurring products, we get the min and max dates
standard_run_recurring <- model_pipeline %>%
  filter(!is.na(ts)) %>%
  group_by(git_id, gitSHA, productStack, branch) %>%
  summarise(
    min_date = min(ts, na.rm = TRUE), 
    max_date = max(ts, na.rm = TRUE),
    runWeek = first(runWeek)
  ) %>%
  ungroup()

# Debug standard_run_recurring to see what columns it has
print("Columns in standard_run_recurring:")
print(colnames(standard_run_recurring))

# Make sure the required columns exist before joining
if(!"branch" %in% colnames(standard_run_recurring)) {
  print("CRITICAL: branch column missing from standard_run_recurring - adding empty one")
  standard_run_recurring$branch <- NA
}

if(!"branch" %in% colnames(model_scan)) {
  print("CRITICAL: branch column missing from model_scan - adding empty one")
  model_scan$branch <- NA
}

print("Final check before critical join:")
print(paste("standard_run_recurring columns:", paste(colnames(standard_run_recurring), collapse=", ")))
print(paste("model_scan columns:", paste(colnames(model_scan), collapse=", ")))

# Simplify and make more direct - use a safer join that specifies all common columns
standard_run_recurring <- standard_run_recurring %>%
  left_join(model_scan, by = c("git_id", "gitSHA", "branch", "runWeek"))

gc() # Add gc call after large join operation

# Verify the results after join
print("Columns in standard_run_recurring after join:")
print(colnames(standard_run_recurring))

# Simplify - use the latest runWeek for processing
latest_runWeek <- scan_periods %>% 
  arrange(desc(startWeek)) %>% 
  slice(1) %>% 
  pull(runWeek)

second_latest_runWeek <- scan_periods %>% 
  arrange(desc(startWeek)) %>% 
  slice(2) %>% 
  pull(runWeek)

print(paste0("Processing primarily for runWeek: ", latest_runWeek))
print(paste0("With comparison to previous runWeek: ", second_latest_runWeek))

# Simplify to focus on just the current and previous runWeeks
# First check if we have the required columns
required_columns <- c("git_id", "gitSHA", "branch", "productStack", "runWeek", "scan_id", "startTime")
missing_columns <- required_columns[!required_columns %in% colnames(standard_run_recurring)]

if(length(missing_columns) > 0) {
  print(paste("CRITICAL: Missing required columns in standard_run_recurring:", paste(missing_columns, collapse=", ")))
  
  # For each missing column, add a placeholder
  for(col in missing_columns) {
    print(paste("Adding empty column for:", col))
    standard_run_recurring[[col]] <- NA
  }
}

# Now proceed with the select
program_semgrep_scan <- standard_run_recurring %>%
  filter(runWeek %in% c(latest_runWeek, second_latest_runWeek)) %>%
  select(git_id, gitSHA, branch, productStack, runWeek, scan_id, startTime) %>%
  unique()

# Debug program_semgrep_scan to see what columns it has
print("Columns in program_semgrep_scan after filtering:")
print(colnames(program_semgrep_scan))

# Make one final check for the branch column
if(!"branch" %in% colnames(program_semgrep_scan)) {
  print("EMERGENCY: branch column missing from program_semgrep_scan at final stage - adding empty one")
  program_semgrep_scan$branch <- NA
}

# Missing scans - focus only on the latest runWeek
missing_scan <- program_semgrep_scan %>% 
  filter(runWeek == latest_runWeek)

# Debug before the final join that's failing
print("Columns in missing_scan before join with gitsha_info:")
print(colnames(missing_scan))
print("Columns in gitsha_info before join:")
print(colnames(gitsha_info))

# Make sure both dataframes have the required columns for joining
join_columns <- c("git_id", "gitSHA", "branch", "productStack")
for(df_name in c("missing_scan", "gitsha_info")) {
  df <- get(df_name)
  missing <- join_columns[!join_columns %in% colnames(df)]
  
  if(length(missing) > 0) {
    print(paste("CRITICAL: Missing join columns in", df_name, ":", paste(missing, collapse=", ")))
    
    # Add missing columns to the dataframe
    for(col in missing) {
      print(paste("Adding empty column", col, "to", df_name))
      df[[col]] <- NA
    }
    
    # Update the dataframe
    assign(df_name, df)
  }
}

# Use an explicit join specification with all matching columns
missing_scan <- missing_scan %>%
  left_join(gitsha_info, by = c("git_id", "gitSHA", "branch", "productStack")) %>%
  filter(is.na(scan_id)) %>%
  ungroup() %>% glimpse()

# ============================ Finalize the data with enrichment ============================

# More memory-efficient joining - only include necessary fields
program_semgrep_scan_attributes <- model_pipeline %>% 
  select(git_id, repository.repoGuid, current_scope, creationDate) %>%
  left_join(model_spr, by = c("repository.repoGuid" = "guid")) %>%
  left_join(mmodel_appdir, by = "deploymentId") %>%
  unique()

# Verify we have all needed data before final join
print("Final data verification before enrichment:")
print(paste("program_semgrep_scan columns:", paste(colnames(program_semgrep_scan), collapse=", ")))
print(paste("program_semgrep_scan_attributes columns:", paste(colnames(program_semgrep_scan_attributes), collapse=", ")))

# Final check to make sure branch is preserved in program_semgrep_scan
if(!"branch" %in% colnames(program_semgrep_scan)) {
  print("CRITICAL: branch lost at final step - adding empty column")
  program_semgrep_scan$branch <- NA
}

# Create a backup of the branch data from program_semgrep_scan
branch_backup <- NULL
if("branch" %in% colnames(program_semgrep_scan)) {
  print("Backing up branch data before final join")
  branch_backup <- program_semgrep_scan %>%
    select(git_id, branch) %>%
    unique()
}

# Final join with more targeted columns
program_semgrep_scan <- program_semgrep_scan %>% 
  left_join(program_semgrep_scan_attributes, by = "git_id") %>%
  glimpse()

# Check if we need to restore branch
if(!"branch" %in% colnames(program_semgrep_scan) && !is.null(branch_backup)) {
  print("Restoring branch data in final output")
  program_semgrep_scan <- program_semgrep_scan %>%
    left_join(branch_backup, by = "git_id")
}

gc() # Add gc call after final join

# ============================ Save Data ============================

# One last validation before saving
print("Final validation before saving:")
for(df_name in c("missing_scan", "program_semgrep_scan")) {
  df <- get(df_name)
  print(paste("Columns in", df_name, ":", paste(colnames(df), collapse=", ")))
}

# If we're still missing branch in either output, add it as NA
for(df_name in c("missing_scan", "program_semgrep_scan")) {
  df <- get(df_name)
  if(!"branch" %in% colnames(df)) {
    print(paste("CRITICAL: Still missing branch in", df_name, "at save step - adding empty column"))
    df$branch <- NA
    assign(df_name, df)
  }
}

write.csv(missing_scan, paste0(data_dir, "/ingest/semgrep_missing_scan_biweekly.csv"), row.names = FALSE)
write.csv(program_semgrep_scan, paste0(data_dir, "/program_semgrep_scan_biweekly.csv"), row.names = FALSE)

# Clean up memory before exiting
rm(list = ls())
gc() # Final garbage collection

print(paste0("Biweekly Semgrep Scan Program Data Ends @ ", Sys.time()))
