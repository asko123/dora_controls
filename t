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
  
  # Get all Mondays between minDate and maxDate
  all_dates <- seq(as.Date(minDate), as.Date(maxDate), by='day')
  mondays_only <- all_dates[weekdays(all_dates) %in% c('Monday')]
  
  # Create a sequence of biweekly Mondays
  # Start with the first Monday and then every 14 days
  biweekly_mondays <- c()
  if(length(mondays_only) > 0) {
    start_monday <- mondays_only[1]
    biweekly_mondays <- seq(from = start_monday, to = maxDate, by = "14 days")
  }
  
  # Format each date with the start time (15:00:00 or 3:00 pm)
  start_dates <- sapply(biweekly_mondays, function(x) paste(as.character(x), '15:00:00', sep=' '))
  
  # Future date for the upcoming period
  if(length(start_dates) > 0) {
    next_date <- as.character(as.Date(biweekly_mondays[length(biweekly_mondays)]) + 14)
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
