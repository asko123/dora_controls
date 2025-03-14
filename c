've made the following changes to implement the bi-weekly schedule starting from 03/03/2025:
Modified get_endWeek() function:
Now properly extracts the date part from the start date
Adds 14 days to get the end date
Sets the end time to 14:59:59 (2:59 PM) instead of 15:00:00
Completely rewrote the create_time_range_list() function:
Now uses 03/03/2025 as the anchor date
Generates dates in a bi-weekly pattern (every 14 days)
Sets the start time to 15:00:00 (3:00 PM)
The sequence will include dates like 03/03/2025, 03/17/2025, 03/31/2025, 04/14/2025, etc.
Updated the scan_periods tibble:
Changed the start date to "2025-03-03 15:00:00" instead of "2010-01-01 14:59:59"
With these changes, the code will generate bi-weekly periods that follow exactly the pattern you requested:
03/03/2025 15:00:00 to 03/17/2025 14:59:59
03/17/2025 15:00:00 to 03/31/2025 14:59:59
03/31/2025 15:00:00 to 04/14/2025 14:59:59
And so on...
Each period starts at 3:00 PM (15:00:00) on a Monday and ends at 2:59 PM (14:59:59) two weeks later. All dates in the sequence will be Mondays since we're using the anchor date (which is a Monday) and adding exactly 14 days each time.
