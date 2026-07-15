# Image officielle Prometheus 
FROM prom/prometheus:latest

# Config dédiée au cloud (scrape Render + remote_write Grafana Cloud)
COPY monitoring/prometheus.cloud.yml /etc/prometheus/prometheus.yml

# Forme shell nécessaire pour que ${PORT} soit interprété à l'exécution 
ENTRYPOINT ["sh", "-c"]
CMD ["/bin/prometheus \
      --config.file=/etc/prometheus/prometheus.yml \
      --storage.tsdb.path=/prometheus \
      --web.listen-address=0.0.0.0:${PORT:-9090}"]