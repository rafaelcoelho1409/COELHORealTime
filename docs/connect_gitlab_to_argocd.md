1) Go to User Settings > Personal access tokens
2) Create a new token (ex: argocd-access) and copy it
3) |
- export GITLAB_TOKEN=<GITLAB_TOKEN>
- argocd login localhost:9080 --insecure
#Insert username and password
- argocd repo add http://localhost:8090/public-projects/COELHORealTime.git \
    --username oauth2 \
    --password $GITLAB_TOKEN \
    --insecure-skip-server-verification \
    --server localhost:9080 \
    --grpc-web