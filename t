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



# If we're still missing required columns in either output, add them as NA
for(df_name in c("missing_scan", "program_semgrep_scan")) {
  df <- get(df_name)
  for(col in c("branch", "guid")) {
    if(!col %in% colnames(df)) {
      print(paste("CRITICAL: Still missing", col, "in", df_name, "at save step - adding empty column"))
      df[[col]] <- NA
      assign(df_name, df)
    }
  }
}
