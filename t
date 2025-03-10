merge_cloud_raw_prev <- read.csv(paste0(data_dir, "/ingest/merge_cloud_branches_semgrep_biweekly.csv"), stringsAsFactors = F)
# Rename "name" column to "branch" for consistency
if("name" %in% colnames(merge_cloud_raw_prev) && !"branch" %in% colnames(merge_cloud_raw_prev)) {
  print("Renaming 'name' column to 'branch' in merge_cloud_raw_prev")
  merge_cloud_raw_prev <- merge_cloud_raw_prev %>% rename(branch = name)
}

merge_prem_raw_prev <- read.csv(paste0(data_dir, "/ingest/merge_prem_branches_semgrep_biweekly.csv"), stringsAsFactors = F)
# Rename "name" column to "branch" for consistency
if("name" %in% colnames(merge_prem_raw_prev) && !"branch" %in% colnames(merge_prem_raw_prev)) {
  print("Renaming 'name' column to 'branch' in merge_prem_raw_prev")
  merge_prem_raw_prev <- merge_prem_raw_prev %>% rename(branch = name)
}

merge_cloud_raw_1 <- read.csv(paste0(data_dir, "/ingest/merge_cloud_branches_result_PART1.csv"), stringsAsFactors = F)
# Rename "name" column to "branch" for consistency
if("name" %in% colnames(merge_cloud_raw_1) && !"branch" %in% colnames(merge_cloud_raw_1)) {
  print("Renaming 'name' column to 'branch' in merge_cloud_raw_1")
  merge_cloud_raw_1 <- merge_cloud_raw_1 %>% rename(branch = name)
}

merge_cloud_raw_2 <- read.csv(paste0(data_dir, "/ingest/merge_cloud_branches_result_PART2.csv"), stringsAsFactors = F)
# Rename "name" column to "branch" for consistency
if("name" %in% colnames(merge_cloud_raw_2) && !"branch" %in% colnames(merge_cloud_raw_2)) {
  print("Renaming 'name' column to 'branch' in merge_cloud_raw_2")
  merge_cloud_raw_2 <- merge_cloud_raw_2 %>% rename(branch = name)
}

merge_prem_raw <- read.csv(paste0(data_dir, "/ingest/merge_prem_branches_result_TP.csv"), stringsAsFactors = F)
# Rename "name" column to "branch" for consistency
if("name" %in% colnames(merge_prem_raw) && !"branch" %in% colnames(merge_prem_raw)) {
  print("Renaming 'name' column to 'branch' in merge_prem_raw")
  merge_prem_raw <- merge_prem_raw %>% rename(branch = name)
}
