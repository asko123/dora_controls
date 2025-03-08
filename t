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
  unique() %>% 
  glimpse()

# Check if scan_periods has any rows
if (nrow(scan_periods) == 0) {
  stop("No valid scan periods found. Please check date ranges and formats.")
}
