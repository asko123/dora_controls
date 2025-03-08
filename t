# Make sure to preserve all needed columns in this join
# First, save a backup of the important columns
print("Before join, model_pipeline has these columns:")
model_pipeline_cols <- colnames(model_pipeline)
print(model_pipeline_cols)

# Save branch column specifically
branch_data <- NULL
if("branch" %in% model_pipeline_cols) {
  print("Saving branch data before join")
  branch_data <- model_pipeline %>% 
    select(git_id, branch) %>% 
    unique()
}

# Perform the join

# Check if branch column was lost in the join
if(!"branch" %in% colnames(model_pipeline) && !is.null(branch_data)) {
  print("WARNING: branch column was lost in the join, adding it back")
  # Add it back
  model_pipeline <- model_pipeline %>%
    left_join(branch_data, by = "git_id")
} else if(!"branch" %in% colnames(model_pipeline)) {
  print("WARNING: branch column wasn't present or couldn't be restored, creating empty one")
  model_pipeline$branch <- NA
}

# Debug model_pipeline after join
print("Columns in model_pipeline after join with model_spr_git (with branch fix):")
print(colnames(model_pipeline))

The issue was that when you joined model_spr_git with model_pipeline, you were doing a left join with model_spr_git on the left side, which means all columns from model_spr_git were preserved, but only the matching columns (based on git_id) from model_pipeline were included. Since the join was only on git_id, other columns like "branch" were potentially lost.
