# =====================================================
# DEFINE STANDARDIZED RUNWEEK FUNCTIONS
# =====================================================

get_endWeek <- function(start_week) {
  # Extract date part from the start_week (removing time component)
  start_date <- as.Date(substr(start_week, 1, 10))
  # Add 14 days to get the end date
  end_date <- start_date + 14
  # Format with time as 14:59:59 (2:59 PM)
  end_week <- paste(as.character(end_date), '14:59:59')
  return(end_week)
}

get_biweekly_scope <- function(startWeek, endWeek) {
  output <- paste('From', startWeek, 'to', endWeek)
  return(output)
}

create_time_range_list <- function(time_range) {
  '''Generate bi-weekly cycles starting from 03/03/2025 as anchor date
     Each cycle starts at 3:00 PM on a Monday and ends at 2:59 PM two weeks later'''
  
  # Parse the input time range
  time_range <- unlist(strsplit(time_range, ','))
  
  # Use 03/03/2025 as anchor date
  anchor_date <- as.Date("2025-03-03")
  
  # Get the latest date (today)
  maxDate <- as.Date(time_range[2])
  
  # Calculate all bi-weekly dates starting from anchor date
  bi_weekly_dates <- seq(from = anchor_date, to = maxDate + 365, by = '14 days')
  
  # Format dates with time component (15:00:00 = 3:00 PM)
  start_dates <- sapply(bi_weekly_dates, function(x) paste(as.character(x), '15:00:00', sep=' '))
  
  result <- paste(start_dates, collapse=';')
  return(result)
}

scan_periods <- tibble(
  # Modified to use our anchor date as the start
  start = c("2025-03-03 15:00:00"),
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
