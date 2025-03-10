# Check if branch column is still missing after attempting to rename
if(!"branch" %in% colnames(merge_cloud_raw)) {
  print("ERROR: branch column missing in merged merge_cloud_raw after rename attempt")
  print("Available columns are:")
  print(paste(colnames(merge_cloud_raw), collapse=", "))
  print("Creating empty branch column as fallback")
  merge_cloud_raw$branch <- NA
}
