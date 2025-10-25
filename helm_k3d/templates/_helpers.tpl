{{/*
Common environment variables for all services
*/}}
{{- define "coelho-realtime.commonEnvVars" -}}
FASTAPI_HOST: "coelho-realtime-fastapi-service"
KAFKA_HOST: "coelho-realtime-kafka-service"
MLFLOW_HOST: "coelho-realtime-mlflow-service"
STREAMLIT_HOST: "coelho-realtime-streamlit-service"
{{- end -}}


{{/*
ConfigMap settings
*/}}
{{- define "coelho-realtime.ConfigMapSettings" -}}
kind: ConfigMap
metadata:
  name: coelho-realtime-{{ .appName }}-configmap
  namespace: coelho
{{- end -}}


{{/*
Deployment settings
*/}}
{{- define "coelho-realtime.DeploymentSettings" -}}
kind: Deployment
metadata:
  name: coelho-realtime-{{ .appName }}-deployment
  namespace: coelho
  labels:
    app: coelho-realtime-{{ .appName }}-deployment
{{- end -}}


{{/*
Service settings
*/}}
{{- define "coelho-realtime.ServiceSettings" -}}
kind: Service
metadata:
  name: coelho-realtime-{{ .appName }}-service
  namespace: coelho
  labels:
    app: coelho-realtime-{{ .appName }}-deployment
spec:
  selector:
    app: coelho-realtime-{{ .appName }}-deployment
{{- end -}}


{{/*
PVC settings
*/}}
{{- define "coelho-realtime.PVCSettings" -}}
kind: PersistentVolumeClaim
metadata:
  name: coelho-realtime-{{ .appName }}-pvc
  namespace: coelho
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: {{ index .root.Values .appName "storageSize" }}
  storageClassName: {{ index .root.Values .appName "storageClassName" }}
{{- end -}}


{{/*
Deployment spec settings
*/}}
{{- define "coelho-realtime.DeploymentSpecSettings" -}}
selector:
  matchLabels:
    app: coelho-realtime-{{ .appName }}-deployment
template:
  metadata:
    labels:
      app: coelho-realtime-{{ .appName }}-deployment
  spec:
    containers:
      - name: coelho-realtime-{{ .appName }}-container
        image: {{ index .root.Values .appName "image" }}
        imagePullPolicy: {{ index .root.Values .appName "imagePullPolicy" }}
        envFrom:
          - configMapRef:
              name: coelho-realtime-{{ .appName }}-configmap
{{- end -}}


{{- define "coelho-realtime.DeploymentResources" -}}
resources:
  requests:
    memory: {{ index .root.Values .appName "resources" "requests" "memory" }}
    cpu: {{ index .root.Values .appName "resources" "requests" "cpu" }}
  limits:
    memory: {{ index .root.Values .appName "resources" "limits" "memory" }}
    cpu: {{ index .root.Values .appName "resources" "limits" "cpu" }}
{{- end -}}