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
