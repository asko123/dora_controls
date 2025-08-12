# DORA Controls Analyzer - Kubernetes Deployment Guide

## 🎯 Current Status: Partially Functional

The Kubernetes deployment is **ready for workbook-only operations** while the full ML pipeline awaits PyTorch 2.6+ release.

## 📋 Prerequisites

### Required
- Kubernetes cluster (v1.20+)
- kubectl configured
- Docker/Container runtime
- 8Gi+ total storage available
- 4+ CPU cores, 8Gi+ RAM recommended

### For GPU Jobs (Optional)
- NVIDIA GPU nodes
- NVIDIA Container Toolkit
- GPU operator installed

## 🚀 Quick Deployment

### Step 1: Build and Deploy Infrastructure
```bash
# Clone and navigate to repository
git clone <repository-url>
cd dora_controls

# Run automated deployment
chmod +x k8s/deploy.sh
./k8s/deploy.sh
```

### Step 2: Upload DORA Legislation
```bash
# Create the DORA legislation ConfigMap (required)
kubectl create configmap dora-legislation-config \
  --from-file=CELEX_32022R2554_EN_TXT.pdf \
  --namespace=dora-analyzer
```

### Step 3: Upload Policy Documents
```bash
# Upload your organization's policy PDFs
kubectl cp policies/ dora-analyzer/$(kubectl get pods -n dora-analyzer -l app=dora-analyzer -o jsonpath='{.items[0].metadata.name}'):/app/policies/
```

## 📊 Available Job Types

### 1. Workbook-Only Analysis (✅ Currently Working)
```bash
kubectl apply -f k8s/job-workbook-only.yaml
```
**Output**: Excel workbooks with DORA domain mappings

### 2. CPU Analysis (⚠️ Limited by PyTorch Issue)
```bash
kubectl apply -f k8s/job-cpu.yaml
```
**Status**: Will fail at ML model initialization

### 3. GPU Analysis (⚠️ Limited by PyTorch Issue)
```bash
kubectl apply -f k8s/job-gpu.yaml
```
**Status**: Will fail at ML model initialization

## 📁 Persistent Storage

The deployment creates three persistent volumes:
- **dora-policies-pvc** (1Gi): Policy document storage
- **dora-output-pvc** (5Gi): Analysis results and reports
- **dora-cache-pvc** (2Gi): Model cache and temporary files

## 🔍 Monitoring and Troubleshooting

### Check Job Status
```bash
# List all jobs
kubectl get jobs -n dora-analyzer

# Check job logs
kubectl logs -f job/dora-workbook-analyzer -n dora-analyzer
kubectl logs -f job/dora-analyzer-cpu -n dora-analyzer
kubectl logs -f job/dora-analyzer-gpu -n dora-analyzer
```

### Check Pod Status
```bash
# List pods
kubectl get pods -n dora-analyzer

# Describe pod for detailed status
kubectl describe pod <pod-name> -n dora-analyzer
```

### Access Results
```bash
# Copy results from completed job
kubectl cp dora-analyzer/<pod-name>:/app/analysis_output/ ./results/
```

## 🛠️ Resource Configuration

### CPU Job Resources
- **Memory**: 4-8Gi (configurable)
- **CPU**: 2-4 cores (configurable)
- **Runtime**: 10-30 minutes for workbook analysis

### GPU Job Resources
- **Memory**: 6-12Gi (configurable)
- **CPU**: 2-4 cores (configurable)
- **GPU**: 1x NVIDIA GPU (Tesla K80 or better)
- **Runtime**: 5-15 minutes for full analysis (when available)

## 🔄 Updating Deployment

### Update Container Images
```bash
# Rebuild and redeploy
docker build -f Dockerfile.cpu -t dora-analyzer:cpu-latest .
docker build -f Dockerfile.gpu -t dora-analyzer:gpu-latest .

# If using registry, push images
docker tag dora-analyzer:cpu-latest your-registry/dora-analyzer:cpu-latest
docker push your-registry/dora-analyzer:cpu-latest
```

### Update Configuration
```bash
# Apply updated ConfigMaps
kubectl apply -f k8s/configmap.yaml

# Restart jobs with new configuration
kubectl delete job dora-workbook-analyzer -n dora-analyzer
kubectl apply -f k8s/job-workbook-only.yaml
```

## 🔐 Security Considerations

### Container Security
- ✅ Non-root user (UID 1000)
- ✅ Read-only root filesystem possible
- ✅ Network policies can be applied
- ✅ Resource limits enforced

### Data Security
- ✅ Persistent volumes for data isolation
- ✅ Namespace isolation
- ✅ ConfigMap for environment variables
- ⚠️ Policy documents require secure upload process

## 📈 Scaling and Performance

### Horizontal Scaling
```bash
# Run multiple workbook jobs in parallel
for i in {1..3}; do
  sed "s/dora-workbook-analyzer/dora-workbook-analyzer-$i/" k8s/job-workbook-only.yaml | kubectl apply -f -
done
```

### Resource Optimization
- **Memory**: Reduce for workbook-only jobs (2Gi sufficient)
- **CPU**: Scale based on document volume
- **Storage**: Increase output PVC for large policy sets

## 🚧 Current Limitations

### PyTorch Security Issue (CVE-2025-32434)
- **Impact**: ML model initialization fails
- **Workaround**: Use workbook-only jobs
- **Resolution**: Pending PyTorch 2.6+ release

### Missing Features in Limited Mode
- ❌ Semantic similarity analysis
- ❌ Zero-shot classification
- ❌ Policy-requirement matching
- ✅ Domain mapping and categorization
- ✅ Excel report generation
- ✅ Basic compliance structure

## 📅 Roadmap

### Phase 1: Current (Workbook Analysis)
- ✅ Infrastructure deployment
- ✅ Workbook generation
- ✅ Domain mapping
- ✅ Excel export

### Phase 2: Full ML Pipeline (Pending PyTorch 2.6+)
- ⏳ Semantic analysis
- ⏳ Policy matching
- ⏳ Compliance scoring
- ⏳ Gap analysis reports

### Phase 3: Advanced Features
- 🔮 Multi-language support
- 🔮 Custom model training
- 🔮 API endpoints
- 🔮 Web dashboard

## 🆘 Support

For deployment issues or questions:
1. Check job logs: `kubectl logs -f job/<job-name> -n dora-analyzer`
2. Verify resource availability: `kubectl describe nodes`
3. Check storage: `kubectl get pvc -n dora-analyzer`
4. Review pod events: `kubectl describe pod <pod-name> -n dora-analyzer`
