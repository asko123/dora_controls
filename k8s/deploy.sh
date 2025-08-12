#!/bin/bash
set -e

echo "Building Docker images..."
docker build -f Dockerfile.cpu -t dora-analyzer:cpu-latest .
docker build -f Dockerfile.gpu -t dora-analyzer:gpu-latest .

echo "Applying Kubernetes manifests..."
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/storage.yaml
kubectl apply -f k8s/dora-legislation-configmap.yaml

echo "Waiting for PVCs to be bound..."
kubectl wait --for=condition=Bound pvc/dora-policies-pvc -n dora-analyzer --timeout=60s
kubectl wait --for=condition=Bound pvc/dora-output-pvc -n dora-analyzer --timeout=60s
kubectl wait --for=condition=Bound pvc/dora-cache-pvc -n dora-analyzer --timeout=60s

echo "Deployment complete!"
echo "To run CPU analysis: kubectl apply -f k8s/job-cpu.yaml"
echo "To run GPU analysis: kubectl apply -f k8s/job-gpu.yaml"
echo ""
echo "Monitor job progress with:"
echo "  kubectl logs -f job/dora-analyzer-cpu -n dora-analyzer"
echo "  kubectl logs -f job/dora-analyzer-gpu -n dora-analyzer"
echo ""
echo "Upload policy files with:"
echo "  kubectl cp policies/ dora-analyzer/\$(kubectl get pods -n dora-analyzer -l app=dora-analyzer -o jsonpath='{.items[0].metadata.name}'):/app/policies/"
echo ""
echo "Retrieve results with:"
echo "  kubectl cp dora-analyzer/\$(kubectl get pods -n dora-analyzer -l app=dora-analyzer -o jsonpath='{.items[0].metadata.name}'):/app/analysis_output/ ./results/"
