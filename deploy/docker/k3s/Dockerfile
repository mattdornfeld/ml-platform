# A self-bootstrapping single-node ML Platform Kubernetes cluster.
FROM rancher/k3s:v1.31.4-k3s1

COPY deploy/kubernetes /opt/ml-platform/kubernetes
COPY deploy/docker/k3s/entrypoint.sh /usr/local/bin/ml-platform-k3s
RUN chmod 755 /usr/local/bin/ml-platform-k3s

ENTRYPOINT ["/usr/local/bin/ml-platform-k3s"]
CMD ["server", "--disable", "traefik"]
