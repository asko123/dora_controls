library(dplyr, quietly = TRUE)
library(lubridate, quietly = TRUE)
library(tidyverse, quietly = TRUE)
library(stringr, quietly = TRUE)
library(sqldf, quietly = TRUE)
# gc() is part of base R, not a separate package

# ============================ READ IN RAW DATA ============================

data_dir <- "/local/data/dev/data" # Sys.getenv("LOCAL_DATA_DIR")
scan_raw <- read.csv(paste0(data_dir, "/model/semgrep_scanner.csv"), stringsAsFactors=FALSE) %>% unique()

# Semgrep branches in previous scope which convey old L1/2
merge_cloud_raw_prev <- read.csv(paste0(data_dir, "/ingest/merge_cloud_branches_semgrep_biweekly.csv"), stringsAsFactors = F)
merge_prem_raw_prev <- read.csv(paste0(data_dir, "/ingest/merge_prem_branches_semgrep_biweekly.csv"), stringsAsFactors = F)
merge_cloud_raw_1 <- read.csv(paste0(data_dir, "/ingest/merge_cloud_branches_result_PART1.csv"), stringsAsFactors = F)
merge_cloud_raw_2 <- read.csv(paste0(data_dir, "/ingest/merge_cloud_branches_result_PART2.csv"), stringsAsFactors = F)
merge_prem_raw <- read.csv(paste0(data_dir, "/ingest/merge_prem_branches_result_TP.csv"), stringsAsFactors = F)
spr_git_mapping_raw_all <- read.csv(paste0(data_dir, "/ingest/spr_mapping_all.csv"), stringsAsFactors = F)
spr_raw <- read.csv(paste0(data_dir, "/ingest/spr.csv"), stringsAsFactors=FALSE) %>% unique()
appdir_raw <- read.csv(paste0(data_dir, "/ingest/appdir.csv"), stringsAsFactors=FALSE) %>% unique()
semgrep_historical <- read.csv(paste0(data_dir, "/ingest/semgrep_scan_historical_biweekly.csv"), stringsAsFactors=FALSE) %>% unique()

# ============================ DEFINE TIME RANGE PARAMETERS ============================

# Set time range - limit to most recent X months (default to 6 months)
months_to_include <- 6  # Adjust as needed (3-6 months suggested)
current_date <- Sys.Date()
cutoff_date <- current_date - months(months_to_include)
cutoff_date_str <- as.character(cutoff_date)

print(paste0("Limiting data processing to the last ", months_to_include, " months (since ", cutoff_date_str, ")"))

# ============================ DEFINE STANDARDIZED RUNWEEK FUNCTIONS ============================

get_startWeek <- function(date_str) {
  # Find the Monday of the week containing the given date
  if (is.na(date_str) || date_str == "" || length(date_str) == 0) {
    return(NA)
  }
  
  # Try to safely parse the date
  tryCatch({
    date <- as.Date(substr(date_str, 1, 10))
    if (is.na(date)) {
      return(NA)
    }
    monday <- date - (wday(date) - 2) %% 7
    return(paste(as.character(monday), '15:00:00'))
  }, error = function(e) {
    # If there's any error in date conversion, return NA
    print(paste("Error processing date:", date_str))
    return(NA)
  })
}

get_endweek <- function(start_week) {
  if (is.na(start_week) || start_week == "" || length(start_week) == 0) {
    return(NA)
  }
  
  # Try to safely parse the date
  tryCatch({
    start_date <- as.Date(substr(start_week, 1, 10))
    if (is.na(start_date)) {
      return(NA)
    }
    end_week <- start_date + 14
    end_week <- paste(as.character(end_week), '15:00:00')
    return(end_week)
  }, error = function(e) {
    # If there's any error in date conversion, return NA
    print(paste("Error processing start week:", start_week))
    return(NA)
  })
}

get_biweekly_scope <- function(startWeek, endWeek) {
  if (is.na(startWeek) || is.na(endWeek) || startWeek == "" || endWeek == "") {
    return(NA)
  }
  output <- paste('From', startWeek, 'to', endWeek)
  return(output)
}

create_time_range_list <- function(time_range) {
  # Break early if input is invalid
  if (is.na(time_range[1]) || is.na(time_range[2])) {
    return(NA)
  }
  
  minDate <- as.Date(time_range[1]) # get the earliest commit date
  maxDate <- as.Date(time_range[2]) # get the latest date (today)
  
  # Apply the cutoff date to limit range
  minDate <- max(minDate, cutoff_date)
  
  all_dates <- seq(as.Date(minDate), as.Date(maxDate), by='day')
  mondays_only <- all_dates[weekdays(all_dates) %in% c('Monday')]
  mondays_only <- sort(mondays_only, decreasing=TRUE)
  dates <- mondays_only[!is.na(mondays_only)]
  new_dates <- as.character(as.Date(dates[1]) + 14)
  start_dates <- as.character(as.Date(dates, '1970-01-01'))
  start_dates <- sapply(start_dates, paste, '14:59:59', sep=' ')
  result <- paste(start_dates, collapse=';')
  return(result)
}

# Define scan periods using the cutoff date
scan_periods <- tibble(
  start = c(cutoff_date_str),
  end = c(as.character(as.POSIXct(Sys.time())))
) 

# Add debug output
print("Date range for scan periods:")
print(scan_periods)

# Create time ranges carefully
scan_periods <- scan_periods %>% 
  mutate(time_range = paste0(start, ' - ', end))

# Add debugging to see what time_range looks like
print("Time range format:")
print(scan_periods$time_range)

# Create time range list more safely
scan_periods <- scan_periods %>%
  mutate(time_range_list = lapply(time_range, function(tr) {
    parts <- strsplit(tr, " - ")[[1]]
    if (length(parts) != 2) {
      print("Invalid time range format")
      return(NA)
    }
    result <- create_time_range_list(c(parts[1], parts[2]))
    return(result)
  }))

# Add debugging for time_range_list
print("Sample time_range_list:")
print(head(scan_periods$time_range_list))

# Process startWeek safely
scan_periods <- scan_periods %>%
  mutate(startWeek = lapply(time_range_list, function(trl) {
    if (is.na(trl)) return(list(NA))
    strsplit(as.character(trl), ";")[[1]]
  })) %>%
  unnest(startWeek) %>%
  ungroup()

# Add debugging for startWeek
print("Sample startWeek values:")
print(head(scan_periods$startWeek))

# Now calculate endWeek and runWeek with NA handling
scan_periods <- scan_periods %>%
  mutate(
    endWeek = sapply(startWeek, function(sw) {
      if (is.na(sw)) return(NA)
      get_endweek(sw)
    }),
    runWeek = mapply(function(sw, ew) {
      if (is.na(sw) || is.na(ew)) return(NA)
      get_biweekly_scope(sw, ew)
    }, startWeek, endWeek)
  ) %>%
  filter(!is.na(runWeek)) %>%  # Remove rows with NA runWeeks
  select(startWeek, endWeek, runWeek) %>% 
  unique()

# Instead, implement a validation function to check if a runWeek's data is usable
validate_runweek_data <- function(runweek_value) {
  # Return TRUE if the runWeek looks valid, FALSE if it appears problematic
  tryCatch({
    # Check if this runWeek exists in model_scan
    runweek_data <- model_scan %>% filter(runWeek == runweek_value)
    if(nrow(runweek_data) == 0) {
      print(paste("WARNING: No scan data for runWeek:", runweek_value, "- skipping"))
      return(FALSE)
    }
    
    # Check required columns in this runWeek's data
    req_cols <- c("git_id", "gitSHA", "branch", "scan_id")
    missing_cols <- req_cols[!req_cols %in% colnames(runweek_data)]
    if(length(missing_cols) > 0) {
      print(paste("WARNING: runWeek", runweek_value, "missing required columns:", 
                  paste(missing_cols, collapse=", "), "- skipping"))
      return(FALSE)
    }
    
    # Check for excessive NA values in key columns
    na_percent <- sum(is.na(runweek_data$branch)) / nrow(runweek_data) * 100
    if(na_percent > 90) {
      print(paste("WARNING: runWeek", runweek_value, "has", round(na_percent, 1), 
                  "% missing branch values - skipping"))
      return(FALSE)
    }
    
    return(TRUE)
  }, error = function(e) {
    print(paste("ERROR processing runWeek", runweek_value, ":", e$message, "- skipping"))
    return(FALSE)
  })
}

# Check if we should keep each runWeek
print("Validating all runWeeks...")
valid_runweeks <- sapply(scan_periods$runWeek, validate_runweek_data)
skipped_runweeks <- scan_periods$runWeek[!valid_runweeks]

if(length(skipped_runweeks) > 0) {
  print(paste("Skipping", length(skipped_runweeks), "problematic runWeeks:"))
  print(skipped_runweeks)
  
  # Keep only valid runWeeks
  scan_periods <- scan_periods %>%
    filter(runWeek %in% scan_periods$runWeek[valid_runweeks])
} else {
  print("All runWeeks passed validation.")
}

scan_periods %>% glimpse()

# Check if scan_periods has any rows
if (nrow(scan_periods) == 0) {
  stop("No valid scan periods found after filtering problematic ones. Please check data.")
}

# Force garbage collection after creating data frames
gc()

# ============================ Model Data ============================

mmodel_appdir <- appdir_raw %>% select(deploymentId,
                                      deploymentName,
                                      ProdAccessPreApproval,
                                      Payment,
                                      External3rdPartyAudit,
                                      FinancialStatement,
                                      DMZ,
                                      HRTP,
                                      SCOE,
                                      ETC,
                                      EQStack,
                                      CCAR,
                                      ProvisionalAudit,
                                      Privacy_New,
                                      OutsourcedHosted,
                                      OutsourcedBuilt,
                                      PublicCloud,
                                      HighRisk,
                                      `MNPI.Data`
                                      ) %>% unique()

model_spr <- spr_raw %>% select(guid, deploymentId, productName, productStatus) %>% unique()

spr_git_mapping_raw_all <- spr_git_mapping_raw_all %>% 
  mutate(current_scope = 0) %>%
  mutate(current_scope = ifelse(is.na(current_scope), 1, 0),
         creationDate = as.POSIXct(creationDate, format= "%a %b %d %H:%M:%S GMT %Y"))

model_spr_git <- spr_git_mapping_raw_all %>% select(guid, repository.repoGuid, stack.name, current_scope, creationDate) %>%
  mutate(git_id = str_extract(repository.repoGuid, '\\d+')) %>% unique()

# Free up memory
rm(spr_git_mapping_raw_all)
gc() # Add gc call after removing large object

# Combine cloud and premise data with time filtering
merge_cloud_raw_prev <- merge_cloud_raw_prev %>% 
  mutate(productStack = 'CLOUDSDLC') %>%
  filter(commit.date >= cutoff_date_str)  # Apply time filter

merge_prem_raw_prev <- merge_prem_raw_prev %>% 
  mutate(productStack = 'GITSDLC') %>%
  filter(commit.date >= cutoff_date_str)  # Apply time filter

merge_raw_prev <- rbind(merge_cloud_raw_prev, merge_prem_raw_prev) %>% unique()

# Free up memory
rm(merge_cloud_raw_prev, merge_prem_raw_prev)
gc() # Add gc call after removing large objects

# Combine merge_cloud_raw_1 and merge_cloud_raw_2 before filtering
merge_cloud_raw <- rbind(merge_cloud_raw_1, merge_cloud_raw_2) %>% 
  mutate(productStack = 'CLOUDSDLC') %>%
  filter(commit.date >= cutoff_date_str)  # Apply time filter

# Free up memory
rm(merge_cloud_raw_1, merge_cloud_raw_2)
gc() # Add gc call after removing large objects

merge_prem_raw <- merge_prem_raw %>% 
  mutate(productStack = 'GITSDLC') %>%
  filter(commit.date >= cutoff_date_str)  # Apply time filter

merge_raw <- rbind(merge_cloud_raw, merge_prem_raw) %>% unique()

# Free up memory
rm(merge_cloud_raw, merge_prem_raw)
gc() # Add gc call after removing large objects

semgrep_historical <- semgrep_historical %>% 
  mutate(git_id = as.character(git_id),
         startTime = scanTime) %>% 
  select(-scanTime) %>%
  filter(startTime >= cutoff_date_str)  # Apply time filter

# semgrep_historical contains future scope. Need to compare to the polaris scan to update those past 'new scope'
new_semgrep_historical <- semgrep_historical %>% 
  select(runWeek, git_id, guid, branch, productStack, scan_id, startTime) %>% 
  unique() %>% glimpse()

rm(semgrep_historical)
gc() # Add gc call after removing large object

# ============================ SCAN DETAILS ============================

# Give credit to a git project if it got scanned once
model_scan <- scan_raw %>% 
  filter(startTime >= cutoff_date_str) %>%  # Apply time filter
  select(git_id, gitSHA, branch, scan_id, startTime) %>% 
  mutate(git_id = as.character(git_id))

model_scan <- sqldf("select a.*, b.startWeek, b.endWeek, b.runWeek 
                    from model_scan a 
                    left join scan_periods b 
                    where a.startTime >= b.startWeek and a.startTime <= b.endWeek")

model_scan <- model_scan %>% 
  group_by(git_id, gitSHA, branch, runWeek) %>%
  summarise(scan_id = max(scan_id), startTime = max(startTime)) %>% 
  ungroup() %>%
  unique()

rm(scan_raw)
gc() # Add gc call after removing large object

# ============================ PRODUCT STACK WITH LATEST COMMIT DATE ============================

# Pipeline data will be used to get the earliest commit date for each git project - This plays the baseline for the standard_run table
model_pipeline <- merge_raw %>% 
  filter(commit.date >= cutoff_date_str) %>%  # Apply time filter
  mutate(git_id = as.character(project_id)) %>%
  mutate(timestamp = gsub("T", " ", commit.date),
         ts = gsub("Z", "", timestamp),
         ts = as.character(as.POSIXct(ts)),
         created_time = gsub("T", " ", commit.created_at),
         created_time = as.character(as.POSIXct(created_time))) %>%
  select(git_id, ts, created_time, branch, default, productStack, gitSHA) %>% 
  unique()

# Debug the ts format before proceeding
print("Sample ts values before processing:")
print(head(model_pipeline$ts))

# Make sure startweek has proper date format before getting endweek
model_pipeline <- model_pipeline %>%
  mutate(
    startweek = get_startWeek(ts)
  )

# Add a check to make sure startweek values are valid
print("Sample startweek values after get_startWeek:")
print(head(model_pipeline$startweek))

# Now calculate endweek only for valid startweek values
model_pipeline <- model_pipeline %>%
  mutate(
    endweek = ifelse(is.na(startweek), NA, get_endweek(startweek)),
    runweek = get_biweekly_scope(startweek, endweek)
  )

# Use a more efficient approach for joining with scan_periods
model_pipeline <- sqldf("select a.*, b.startWeek, b.endWeek, b.runWeek 
                        from model_pipeline a 
                        left join scan_periods b 
                        where a.ts >= b.startWeek and a.ts <= b.endWeek")

# Skip problematic runWeeks in model_pipeline if they exist
if("runWeek" %in% colnames(model_pipeline) && length(skipped_runweeks) > 0) {
  model_pipeline <- model_pipeline %>%
    filter(!runWeek %in% skipped_runweeks)
}

# Debug model_pipeline to see what columns it has
print("Columns in model_pipeline after SQL join:")
print(colnames(model_pipeline))

rm(merge_raw)
gc() # Add gc call after removing large object

gitsha_info <- model_pipeline %>% select(git_id, branch, default, productStack, gitSHA) %>% unique()

# Debug gitsha_info to see what columns it has
print("Columns in gitsha_info:")
print(colnames(gitsha_info))

# Make sure to preserve all needed columns in this join
# First, save a backup of the important columns
print("Before join, model_pipeline has these columns:")
model_pipeline_cols <- colnames(model_pipeline)
print(model_pipeline_cols)

# Save branch column specifically
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



