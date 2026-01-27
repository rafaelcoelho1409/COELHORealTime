"""
COELHO RealTime — Platform Architecture Diagram

Generates a comprehensive architecture visualization of the COELHO RealTime
ML platform using the Python `diagrams` library (mingrammer/diagrams).

Usage:
    python main.py

Output:
    coelho_realtime_architecture.png
"""

import os
from urllib.request import urlretrieve

from diagrams import Diagram, Cluster, Edge
from diagrams.custom import Custom
from diagrams.k8s.ecosystem import Helm
from diagrams.onprem.container import Docker
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.gitops import ArgoCD
from diagrams.onprem.iac import Terraform
from diagrams.onprem.inmemory import Redis
from diagrams.onprem.monitoring import Grafana, Prometheus
from diagrams.onprem.queue import Kafka
from diagrams.onprem.vcs import Gitlab
from diagrams.programming.language import Python

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)


# ── Custom Icon Downloads ─────────────────────────────────────

CUSTOM_ICONS = [
    ("https://k3d.io/stable/static/img/k3d_logo_black_blue.svg", "k3d.svg"),
    (
        "https://awsmp-logos.s3.amazonaws.com/c6cbc20d-0d79-4256-bbd1-dbb8d219945b/"
        "8433cadab9c1c3920e06e695055fb019.png",
        "minio.png",
    ),
    (
        "https://spark.apache.org/images/spark-logo-trademark.png",
        "spark.png",
    ),
    (
        "https://avatars.githubusercontent.com/u/49767398?s=200&v=4",
        "delta_lake.png",
    ),
    (
        "https://icon.icepanel.io/Technology/svg/Svelte.svg",
        "svelte.png",
    ),
    (
        "https://avatars.githubusercontent.com/u/39938107?s=200&v=4",
        "mlflow.png",
    ),
    (
        "https://avatars.githubusercontent.com/u/156354296?s=200&v=4",
        "fastapi.png",
    ),
    ("https://duckdb.org/images/logo-dl/DuckDB_Logo.png", "duckdb.png"),
    ("https://skaffold.dev/images/skaffold-logo-white.png", "skaffold.png"),
    (
        "https://rancher.com/docs/img/logo-square.png",
        "rancher.png",
    ),
]

print("Downloading custom icons...")
for url, filename in CUSTOM_ICONS:
    if not os.path.exists(filename):
        try:
            urlretrieve(url, filename)
            print(f"  OK  {filename}")
        except Exception as e:
            print(f"  FAIL {filename}: {e}")
    else:
        print(f"  --  {filename} (cached)")


# ── Cluster Style Helpers ─────────────────────────────────────


def layer_style(bgcolor, pencolor):
    return {
        "bgcolor": bgcolor,
        "pencolor": pencolor,
        "penwidth": "4",
        "style": "rounded",
        "margin": "30",
        "fontsize": "18",
        "fontcolor": "#333333",
    }


def sub_style(bgcolor="#FFFFFF", pencolor="#999999"):
    return {
        "bgcolor": bgcolor,
        "pencolor": pencolor,
        "penwidth": "2",
        "style": "rounded,dashed",
        "margin": "20",
        "fontsize": "14",
        "fontcolor": "#666666",
    }


# ── Build Diagram ─────────────────────────────────────────────

with Diagram(
    "COELHO RealTime \u2014 Platform Architecture",
    show=False,
    filename="coelho_realtime_architecture",
    outformat="png",
    direction="LR",
    graph_attr={
        "fontsize": "28",
        "bgcolor": "white",
        "pad": "2.5",
        "splines": "curved",
        "nodesep": "1.0",
        "ranksep": "2.5",
        "compound": "true",
        "dpi": "200",
    },
):

    # ── Layer 1: Container Orchestration ──────────────────────
    with Cluster(
        "Container Orchestration & Infrastructure as Code",
        graph_attr=layer_style("#E6E6FA", "#5C4EE5"),
    ):
        tf = Terraform("Terraform")
        docker = Docker("Docker")
        k3d = Custom("K3D Cluster\n(1 Server + 3 Agents)", "k3d.svg")
        rancher = Custom("Rancher\n(Cluster Management)", "rancher.png")
        k3d_registry = Custom("K3D Registry\n(localhost:5000)", "k3d.svg")
        helm = Helm("Helm Charts")
        skaffold = Custom("Skaffold\n(Dev Hot-Reload)", "skaffold.png")

    # ── Layer 2: GitOps & CI/CD ───────────────────────────────
    with Cluster(
        "GitOps & CI/CD Layer",
        graph_attr=layer_style("#FFF4E6", "#FF6B35"),
    ):
        gitlab = Gitlab("GitLab\n(CI/CD Pipelines)")
        argocd = ArgoCD("ArgoCD\n(GitOps Sync)")
        argocd_updater = ArgoCD("ArgoCD\nImage Updater")

    # ── Layer 3: Data & Streaming ─────────────────────────────
    with Cluster(
        "Data & Streaming Layer",
        graph_attr=layer_style("#F0FFF0", "#28A745"),
    ):
        kafka_producers = Python("Kafka Producers\n(3 Topics)")
        kafka = Kafka("Apache Kafka\n(KRaft Mode)")

        with Cluster(
            "Spark Structured Streaming",
            graph_attr=sub_style("#F8FFF8", "#28A745"),
        ):
            spark_tfd = Custom("TFD\n(Transaction\nFraud Detection)", "spark.png")
            spark_eta = Custom("ETA\n(Estimated Time\nof Arrival)", "spark.png")
            spark_ecci = Custom("ECCI\n(E-Commerce Customer\nInteractions)", "spark.png")

        delta_lake = Custom("Delta Lake\n(Lakehouse)", "delta_lake.png")

    # ── Layer 4: Application Services ─────────────────────────
    with Cluster(
        "Application Services Layer",
        graph_attr=layer_style("#E8F4F8", "#0066CC"),
    ):
        with Cluster(
            "FastAPI \u2014 Unified ML Service",
            graph_attr=sub_style("#F0F4FF", "#0066CC"),
        ):
            fastapi_inc = Custom("/v1/incremental\n(River ML)", "fastapi.png")
            fastapi_batch = Custom(
                "/v1/batch\n(Scikit-Learn + CatBoost)", "fastapi.png"
            )
            fastapi_sql = Custom("/v1/sql\n(DuckDB Queries)", "fastapi.png")

        sveltekit = Custom("SvelteKit\n(Frontend)", "svelte.png")
        mlflow = Custom("MLflow\n(Experiment Tracking)", "mlflow.png")
        redis = Redis("Redis\n(Model Cache\n30s TTL)")
        duckdb = Custom("DuckDB\n(SQL Engine)", "duckdb.png")
        minio = Custom("MinIO\n(Object Storage)", "minio.png")
        postgresql = PostgreSQL("PostgreSQL\n(MLflow Metadata)")

    # ── Layer 5: Observability & Monitoring ───────────────────
    with Cluster(
        "Observability & Monitoring",
        graph_attr=layer_style("#FFF0F0", "#E53E3E"),
    ):
        prometheus = Prometheus("Prometheus\n(ServiceMonitors)")
        grafana = Grafana("Grafana\n(Dashboards)")
        alertmanager = Prometheus("Alertmanager")
        karma = Prometheus("Karma\n(Alert Dashboard)")

    # ===========================================================
    # EDGES
    # ===========================================================

    # ── Invisible edges to enforce horizontal layer ordering ──
    # Multiple anchors per layer pair for stronger constraint
    # Layer 1 → Layer 2
    helm >> Edge(style="invis", weight="10") >> gitlab
    k3d >> Edge(style="invis", weight="10") >> argocd
    k3d_registry >> Edge(style="invis", weight="10") >> argocd_updater
    # Layer 2 → Layer 3
    gitlab >> Edge(style="invis", weight="10") >> kafka_producers
    argocd >> Edge(style="invis", weight="10") >> kafka
    # Layer 3 → Layer 4
    kafka >> Edge(style="invis", weight="10") >> sveltekit
    delta_lake >> Edge(style="invis", weight="10") >> fastapi_inc
    spark_eta >> Edge(style="invis", weight="10") >> mlflow
    # Layer 4 → Layer 5
    minio >> Edge(style="invis", weight="10") >> prometheus
    postgresql >> Edge(style="invis", weight="10") >> grafana
    redis >> Edge(style="invis", weight="10") >> alertmanager

    # ── Layer 1: Infrastructure Provisioning ──────────────────
    tf >> Edge(color="#5C4EE5", style="bold", penwidth="3") >> docker
    docker >> Edge(color="#0066CC", style="bold", penwidth="3") >> k3d
    tf >> Edge(color="#5C4EE5", style="bold", penwidth="2",
              label="provisions") >> rancher
    rancher >> Edge(color="#5C4EE5", style="bold", penwidth="2",
                    label="manages") >> k3d
    docker >> Edge(color="#0066CC", style="dashed", penwidth="2") >> k3d_registry
    k3d >> Edge(color="#0066CC", style="bold", penwidth="3") >> helm
    skaffold >> Edge(
        color="#0066CC", style="dashed", penwidth="2",
        label="dev\nworkflow",
    ) >> helm

    # ── Helm Deployment Edges (solid dark blue) ───────────────
    # Helm deploys all microservices as subchart dependencies
    # constraint=false prevents these cross-layer edges from disrupting layout
    _helm_edge = dict(color="#004D9966", style="dashed", penwidth="1.0",
                      constraint="false")

    # Data & Streaming Layer
    helm >> Edge(**_helm_edge) >> kafka_producers
    helm >> Edge(**_helm_edge) >> kafka
    helm >> Edge(**_helm_edge) >> spark_tfd
    helm >> Edge(**_helm_edge) >> spark_eta
    helm >> Edge(**_helm_edge) >> spark_ecci

    # Application Services Layer
    helm >> Edge(**_helm_edge) >> fastapi_inc
    helm >> Edge(**_helm_edge) >> fastapi_batch
    helm >> Edge(**_helm_edge) >> fastapi_sql
    helm >> Edge(**_helm_edge) >> sveltekit
    helm >> Edge(**_helm_edge) >> mlflow
    helm >> Edge(**_helm_edge) >> redis
    helm >> Edge(**_helm_edge) >> minio
    helm >> Edge(**_helm_edge) >> postgresql

    # Observability & Monitoring Layer
    helm >> Edge(**_helm_edge) >> prometheus
    helm >> Edge(**_helm_edge) >> grafana
    helm >> Edge(**_helm_edge) >> alertmanager
    helm >> Edge(**_helm_edge) >> karma

    # ── GitOps Flow (dashed orange) ───────────────────────────
    argocd >> Edge(
        color="#FF6B35", style="dashed", penwidth="2",
        label="watches\nrepo (CD)", constraint="false",
    ) >> gitlab

    gitlab >> Edge(
        color="#FF6B35", style="dashed", penwidth="2",
        label="CI/CD\nimage push", constraint="false",
    ) >> k3d_registry

    k3d_registry >> Edge(
        color="#FF6B35", style="dashed", penwidth="2",
        label="detect\nnew tags",
    ) >> argocd_updater

    argocd_updater >> Edge(
        color="#FF6B35", style="dashed", penwidth="2",
    ) >> argocd

    argocd >> Edge(
        color="#FF6B35", style="dashed", penwidth="2",
        label="GitOps\ndeploy", constraint="false",
    ) >> helm

    # ── Data Pipeline (bold green) ────────────────────────────
    kafka_producers >> Edge(
        color="#28A745",
        style="bold",
        penwidth="3",
        label="produce\nevents",
    ) >> kafka

    kafka >> Edge(color="#28A745", style="bold", penwidth="3") >> spark_tfd
    kafka >> Edge(color="#28A745", style="bold", penwidth="2") >> spark_eta
    kafka >> Edge(color="#28A745", style="bold", penwidth="2") >> spark_ecci

    spark_tfd >> Edge(
        color="#28A745",
        style="bold",
        penwidth="3",
        label="persist to\nlakehouse",
    ) >> delta_lake
    spark_eta >> Edge(color="#28A745", style="bold", penwidth="2") >> delta_lake
    spark_ecci >> Edge(color="#28A745", style="bold", penwidth="2") >> delta_lake

    delta_lake >> Edge(
        color="#28A745", style="bold", penwidth="2",
        label="stored on", constraint="false",
    ) >> minio

    # ── Application Data Flow (bold blue) ─────────────────────
    kafka >> Edge(
        color="#0066CC",
        style="bold",
        penwidth="2",
        label="incremental\nlearning",
    ) >> fastapi_inc

    minio >> Edge(
        color="#0066CC",
        style="bold",
        penwidth="2",
        label="batch\ndata reads",
    ) >> fastapi_batch

    fastapi_sql >> Edge(color="#0066CC", style="bold", penwidth="2") >> duckdb

    duckdb >> Edge(
        color="#0066CC", style="dashed", penwidth="2",
        label="query\nDelta Lake", constraint="false",
    ) >> minio

    # FastAPI <-> Redis (model cache)
    fastapi_inc >> Edge(
        color="#0066CC",
        style="bold",
        penwidth="2",
        label="cache\nmodels",
    ) >> redis
    fastapi_batch >> Edge(color="#0066CC", style="bold", penwidth="2") >> redis

    # FastAPI -> MLflow (experiment logging)
    fastapi_inc >> Edge(
        color="#6B46C1",
        style="bold",
        penwidth="2",
        label="log\nexperiments",
    ) >> mlflow
    fastapi_batch >> Edge(color="#6B46C1", style="bold", penwidth="2") >> mlflow

    # MLflow backends
    mlflow >> Edge(
        color="#6B46C1",
        style="bold",
        penwidth="2",
        label="metadata",
    ) >> postgresql
    mlflow >> Edge(
        color="#6B46C1",
        style="bold",
        penwidth="2",
        label="artifacts",
    ) >> minio

    # SvelteKit -> FastAPI (frontend API calls)
    sveltekit >> Edge(
        color="#00B4D8",
        style="bold",
        penwidth="3",
        label="REST API",
    ) >> fastapi_inc
    sveltekit >> Edge(color="#00B4D8", style="bold", penwidth="2") >> fastapi_batch
    sveltekit >> Edge(color="#00B4D8", style="bold", penwidth="2") >> fastapi_sql

    # ── Monitoring Flow (dotted red) ──────────────────────────
    prometheus >> Edge(
        color="#E53E3E",
        style="dotted",
        penwidth="2",
        label="dashboards",
    ) >> grafana
    prometheus >> Edge(
        color="#E53E3E",
        style="dotted",
        penwidth="2",
        label="alerts",
    ) >> alertmanager
    alertmanager >> Edge(
        color="#E53E3E",
        style="dotted",
        penwidth="2",
        label="dashboard",
        constraint="false",
    ) >> karma

    # Scraping edges (key services only, constraint=false to avoid layout disruption)
    fastapi_inc >> Edge(
        color="#E53E3E", style="dotted", penwidth="1",
        label="metrics", constraint="false",
    ) >> prometheus
    kafka >> Edge(
        color="#E53E3E", style="dotted", penwidth="1", constraint="false",
    ) >> prometheus
    spark_tfd >> Edge(
        color="#E53E3E", style="dotted", penwidth="1", constraint="false",
    ) >> prometheus


print("\nDiagram generated: coelho_realtime_architecture.png")
