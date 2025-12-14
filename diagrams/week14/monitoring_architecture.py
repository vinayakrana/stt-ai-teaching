"""
Generate monitoring architecture diagram for Week 14: Model Monitoring
Replaces ASCII art with professional diagrams library visualization
"""
from diagrams import Diagram, Cluster, Edge
from diagrams.custom import Custom
from diagrams.onprem.monitoring import Prometheus, Grafana
from diagrams.onprem.database import Influxdb
from diagrams.onprem.workflow import Airflow
from diagrams.aws.storage import S3
from diagrams.programming.framework import FastAPI

def generate_monitoring_architecture():
    """Create professional monitoring architecture using diagrams library."""

    with Diagram("ML Monitoring Architecture",
                 filename="../figures/week14_monitoring_architecture",
                 show=False,
                 direction="TB",
                 graph_attr={"bgcolor": "white"}):

        # Prediction API
        api = FastAPI("Prediction API")

        # Logging layer
        with Cluster("Data Collection"):
            logger = Custom("Async Logger", "")

        # Storage layer
        with Cluster("Storage Layer"):
            storage = S3("Object Storage\n(S3/GCS)")

        # ETL Processing
        with Cluster("Data Processing"):
            etl = Airflow("ETL Pipeline\n(Airflow/dbt)\nDaily/Hourly")

        # Analysis layer
        with Cluster("Drift Detection"):
            drift = Custom("Drift Analysis\n(Evidently)", "")

        # Metrics and monitoring
        with Cluster("Monitoring Stack"):
            metrics_db = Prometheus("Metrics DB\n(Prometheus/InfluxDB)")
            dashboard = Grafana("Dashboard\n(Grafana)")

        # Alerting
        with Cluster("Alerting"):
            alerts = Custom("Alerts\n(PagerDuty/Slack)", "")

        # Flow
        api >> Edge(label="log predictions") >> logger
        logger >> storage
        storage >> etl
        etl >> drift
        drift >> metrics_db
        metrics_db >> dashboard
        dashboard >> alerts

    print("Generated: figures/week14_monitoring_architecture.png")

def generate_monitoring_architecture_graphviz():
    """Fallback: Create monitoring architecture using graphviz."""
    from graphviz import Digraph

    dot = Digraph('Monitoring Architecture',
                  graph_attr={'rankdir': 'TB', 'bgcolor': 'white'})

    # Define nodes
    dot.node('API', 'Prediction API', shape='box', style='filled', fillcolor='lightblue')
    dot.node('Logger', 'Logger\n(Async)', shape='box', style='filled', fillcolor='lightyellow')
    dot.node('Storage', 'Object Storage\n(S3, GCS)', shape='cylinder', style='filled', fillcolor='lightgray')
    dot.node('ETL', 'ETL Pipeline\n(Airflow, dbt)\nDaily/Hourly', shape='box', style='filled', fillcolor='lightgreen')
    dot.node('Drift', 'Drift Analysis\n(Evidently, custom)', shape='box', style='filled', fillcolor='lightcoral')
    dot.node('MetricsDB', 'Metrics Database\n(Prometheus, InfluxDB)', shape='cylinder', style='filled', fillcolor='lightpink')
    dot.node('Dashboard', 'Dashboard\n(Grafana, Tableau)', shape='box', style='filled', fillcolor='lightyellow')
    dot.node('Alerts', 'Alerts\n(PagerDuty, Slack)', shape='box', style='filled', fillcolor='lightcoral')

    # Flow
    dot.edge('API', 'Logger', label='log predictions')
    dot.edge('Logger', 'Storage', label='store')
    dot.edge('Storage', 'ETL', label='scheduled')
    dot.edge('ETL', 'Drift', label='analyze')
    dot.edge('Drift', 'MetricsDB', label='save metrics')
    dot.edge('MetricsDB', 'Dashboard', label='visualize')
    dot.edge('Dashboard', 'Alerts', label='trigger')

    # Save diagram
    dot.render('../figures/week14_monitoring_architecture_graphviz', format='png', cleanup=True)
    print("Generated: figures/week14_monitoring_architecture_graphviz.png")

if __name__ == '__main__':
    print("Generating Week 14 monitoring architecture diagram...")

    # Try diagrams library first (more professional)
    try:
        generate_monitoring_architecture()
    except Exception as e:
        print(f"Note: diagrams library not available: {e}")
        print("Falling back to graphviz...")
        generate_monitoring_architecture_graphviz()

    print("Week 14 diagram generated successfully!")
