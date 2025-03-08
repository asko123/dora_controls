# Debug standard_run_recurring to see what columns it has
print("Columns in standard_run_recurring:")
print(colnames(standard_run_recurring))

# Make sure the required columns exist before joining
if(!"branch" %in% colnames(standard_run_recurring) || !"branch" %in% colnames(model_scan)) {
  print("WARNING: Required columns missing before join:")
  print(paste("standard_run_recurring columns:", paste(colnames(standard_run_recurring), collapse=", ")))
  print(paste("model_scan columns:", paste(colnames(model_scan), collapse=", ")))
  
  # Fix columns if needed
  if(!"branch" %in% colnames(model_scan)) {
    print("Adding empty branch column to model_scan")
    model_scan$branch <- NA
  }
}
