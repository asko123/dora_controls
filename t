# =====================================================
# DEFINE STANDARDIZED RUNWEEK FUNCTIONS
# =====================================================

get_endWeek <- function(start_week) {
  # Extract date part only from start_week
  start_date <- as.Date(substr(start_week, 1, 10))
  # Add 14 days to get to the end date
  end_date <- start_date + 14
  # Format with 14:59:59 (2:59 pm) time
  end_week <- paste(as.character(end_date), '14:59:59')
  return(end_week)
}

get_biweekly_scope <- function(startWeek, endWeek) {
  output <- paste('From', startWeek, 'to', endWeek)
  return(output)
}

create_time_range_list <- function(time_range) {
  # time_range: applied on output of round_start_time_period (the assigned Monday)
  time_range <- unlist(strsplit(time_range, ','))
  minDate <- as.Date(time_range[1])  # get the earliest commit date
  maxDate <- as.Date(time_range[2])  # get the latest date (today)
  
  # Use a reference date that we know is the start of a biweekly period
  reference_date <- as.Date("2025-02-17")  # This is a known Monday start date
  
  # Calculate all biweekly Mondays by extending backwards and forwards from the reference date
  days_diff <- as.numeric(difftime(maxDate, reference_date, units = "days"))
  min_days_diff <- as.numeric(difftime(reference_date, minDate, units = "days"))
  
  # Generate dates forward from reference date
  forward_periods <- floor(days_diff / 14) + 1
  forward_dates <- reference_date + (0:(forward_periods)) * 14
  
  # Generate dates backward from reference date
  backward_periods <- floor(min_days_diff / 14) + 1
  backward_dates <- reference_date - (1:backward_periods) * 14
  
  # Combine all dates and filter to those within our range
  all_biweekly_dates <- c(backward_dates, reference_date, forward_dates)
  biweekly_mondays <- all_biweekly_dates[all_biweekly_dates >= minDate & all_biweekly_dates <= maxDate]
  
  # Sort dates
  biweekly_mondays <- sort(biweekly_mondays)
  
  # Format each date with the start time (15:00:00 or 3:00 pm)
  start_dates <- sapply(biweekly_mondays, function(x) paste(as.character(x), '15:00:00', sep=' '))
  
  # Include one future period
  if(length(biweekly_mondays) > 0) {
    next_date <- as.character(max(biweekly_mondays) + 14)
    next_period <- paste(next_date, '15:00:00', sep=' ')
    start_dates <- c(start_dates, next_period)
  }
  
  result <- paste(start_dates, collapse=';')
  return(result)
}

# Function to get date from 12 months ago
get_twelve_months_ago <- function() {
  current_date <- Sys.Date()
  twelve_months_ago <- current_date - 365  # Approximate 12 months
  return(format(twelve_months_ago, "%Y-%m-%d 14:59:59"))
}

scan_periods <- tibble(
  # Dynamically set start date to 12 months ago
  start = c(get_twelve_months_ago()),
  end = c(as.character(as.POSIXct(Sys.time(), tz="UTC")))
) %>% mutate(time_range = paste0(start, ',', end)) %>%
  mutate(time_range_list = lapply(time_range, create_time_range_list),
         startWeek = strsplit(as.character(time_range_list), ";")) %>%
  ungroup() %>%
  unnest(startWeek) %>%
  mutate(endWeek = get_endWeek(startWeek)) %>%
  mutate(runWeek = get_biweekly_scope(startWeek, endWeek)) %>%
  select(startWeek, endWeek, runWeek) %>%
  glimpse()
