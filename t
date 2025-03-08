# Missing scans - focus only on the latest runWeek
missing_scan <- program_semgrep_scan %>% 
  filter(runWeek == latest_runWeek)

# Debug before the final join that's failing
print("Columns in missing_scan before join with gitsha_info:")
print(colnames(missing_scan))
print("Columns in gitsha_info before join:")
print(colnames(gitsha_info))

# Use an explicit join specification with all matching columns
missing_scan <- missing_scan %>%
  left_join(gitsha_info, by = c("git_id", "gitSHA", "branch", "productStack")) %>%
  filter(is.na(scan_id)) %>%
  ungroup() %>% glimpse()
