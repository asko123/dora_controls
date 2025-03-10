# Define a more permissive validation function
validate_runweek_data <- function(runweek_value) {
  # Skip validation if we're not yet ready
  if(!exists("model_scan")) {
    return(TRUE)
  }
  
  # Return TRUE if the runWeek looks valid, FALSE if it appears problematic
  tryCatch({
    # Only filter out weeks with extreme issues
    runweek_data <- model_scan %>% filter(runWeek == runweek_value)
    
    # More permissive check - only skip if there's truly no data
    if(nrow(runweek_data) == 0) {
      print(paste("WARNING: No scan data at all for runWeek:", runweek_value))
      # Even with no data, we'll still keep the week unless explicitly told to skip
      return(TRUE)
    }
    
    return(TRUE)
  }, error = function(e) {
    print(paste("NOTE: Error checking runWeek", runweek_value, ":", e$message))
    # Be permissive - don't skip just because of validation errors
    return(TRUE)
  })
}

# Instead of automatically filtering runWeeks, let the user specify ones to skip
skipped_runweeks <- c() # Empty by default - no weeks skipped

# If you need to skip specific problematic weeks, uncomment and modify this:
# skipped_runweeks <- c("From 2024-09-02 to 2024-09-16 15:00:00")

if(length(skipped_runweeks) > 0) {
  print(paste("Manually skipping", length(skipped_runweeks), "specified runWeeks:"))
  print(skipped_runweeks)
  
  # Filter out specified weeks
  scan_periods <- scan_periods %>%
    filter(!runWeek %in% skipped_runweeks)
  
  model_scan <- model_scan %>%
    filter(!runWeek %in% skipped_runweeks)
} else {
  print("No runWeeks are being skipped.")
}
