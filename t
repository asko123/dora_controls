# Check if branch column is still missing after attempting to rename
if(!"branch" %in% colnames(merge_cloud_raw_prev)) {
  print("ERROR: branch column missing in merge_cloud_raw_prev after rename attempt")
  print("Available columns are:")
  print(paste(colnames(merge_cloud_raw_prev), collapse=", "))
  print("Creating empty branch column as fallback")
  merge_cloud_raw_prev$branch <- NA
}
