{{/*
Generate image name based on environment
Usage: {{ include "coelho-realtime.imageName" (dict "appName" "fastapi" "root" .) }}
*/}}
{{- define "coelho-realtime.imageName" -}}
{{- $image := index .root.Values .appName "image" -}}
{{- if eq .root.Values.environment "production" -}}
  {{- if not (hasPrefix .root.Values.registry.url $image) -}}
    {{- printf "%s/%s" .root.Values.registry.url $image -}}
  {{- else -}}
    {{- $image -}}
  {{- end -}}
{{- else -}}
  {{- $image -}}
{{- end -}}
{{- end -}}


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
  namespace: {{ .root.Release.Namespace }}
{{- end -}}


{{/*
Deployment settings
*/}}
{{- define "coelho-realtime.DeploymentSettings" -}}
kind: Deployment
metadata:
  name: coelho-realtime-{{ .appName }}-deployment
  namespace: {{ .root.Release.Namespace }}
  labels:
    app.kubernetes.io/name: {{ .root.Chart.Name }}
    app.kubernetes.io/instance: {{ .root.Release.Name }}
    app.kubernetes.io/version: {{ .root.Chart.AppVersion }}
    app.kubernetes.io/component: {{ .appName }}
    app.kubernetes.io/managed-by: {{ .root.Release.Service }}
{{- end -}}


{{/*
Service settings
*/}}
{{- define "coelho-realtime.ServiceSettings" -}}
kind: Service
metadata:
  name: coelho-realtime-{{ .appName }}-service
  namespace: {{ .root.Release.Namespace }}
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
  namespace: {{ .root.Release.Namespace }}
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
    {{- if eq .root.Values.environment "production" }}
    imagePullSecrets:
      - name: {{ .root.Values.registry.imagePullSecret }}
    {{- end }}
    #securityContext:
    #  runAsNonRoot: true
    #  runAsUser: 1000
    #  fsGroup: 1000
    containers:
      - name: coelho-realtime-{{ .appName }}-container
        image: {{ include "coelho-realtime.imageName" (dict "appName" .appName "root" .root) }}
        imagePullPolicy: {{ index .root.Values .appName "imagePullPolicy" }}
        #securityContext:
        #  allowPrivilegeEscalation: false
        #  capabilities:
        #    drop:
        #      - ALL
        #  readOnlyRootFilesystem: false
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