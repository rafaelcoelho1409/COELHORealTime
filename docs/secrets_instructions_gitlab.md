# Uploading secrets securely on GitLab

1. Go to Preferences > Access Tokens and create a new API key
2. Set the following permissions for this new API key: api, read_api, read_repository, write_repository
3. Copy the token and paste on GITLAB_TOKEN into .env file
4. run "python3 upload_env_to_gitlab.py" to upload secrets to CI/CD variables (to avoid exposing them publicly)
5. Go to Settings > Repository > Protected branches, add the "master" branch (or the current branch you are working on) and set "Allowed to merge" and "Allowed to push and merge" options to "Developers + Maintainers". It makes GitLab Runner access all internal variables correctly

## Important observations
- Centralize all secrets into a .env file on the project root folder in order to upload them to GitLab by using upload_env_to_gitlab.py file. After this, distribute the necessary secrets into each Kubernetes deployments