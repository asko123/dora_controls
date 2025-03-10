# Check if branch column is still missing after attempting to rename
if(!"branch" %in% colnames(merge_prem_raw)) {
  print("ERROR: branch column missing in merge_prem_raw after rename attempt")
  print("Available columns are:")
  print(paste(colnames(merge_prem_raw), collapse=", "))
  print("Creating empty branch column as fallback")
  merge_prem_raw$branch <- NA
}
