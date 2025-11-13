mirrors:
  "localhost:${registry_port}":
    endpoint:
      - "http://${registry_name}:5000"
  "${registry_name}:5000":
    endpoint:
      - "http://${registry_name}:5000"
configs:
  "localhost:${registry_port}":
    tls:
      insecure_skip_verify: true
  "${registry_name}:5000":
    tls:
      insecure_skip_verify: true
