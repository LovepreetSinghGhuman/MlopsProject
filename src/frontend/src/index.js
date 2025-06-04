/BERTopic-Azure-ML-Deployment
│
├── /backend
│   ├── /app
│   │   ├── main.py                # FastAPI application
│   │   ├── model.py               # Model training and inference logic
│   │   ├── requirements.txt        # Python dependencies
│   │   └── Dockerfile              # Dockerfile for FastAPI app
│   ├── /k8s
│   │   ├── deployment.yaml         # Kubernetes deployment configuration
│   │   ├── service.yaml            # Kubernetes service configuration
│   │   └── ingress.yaml            # Ingress configuration for NGINX
│   └── /azure
│       ├── azure_ml.py             # Azure ML training script
│       └── azure_config.json       # Azure ML configuration
│
├── /frontend
│   ├── /src
│   │   ├── App.js                  # Main React component
│   │   ├── index.js                # Entry point for React app
│   │   └── package.json             # Frontend dependencies
│   └── Dockerfile                   # Dockerfile for React app
│
├── /nginx
│   ├── nginx.conf                   # NGINX configuration
│   └── Dockerfile                   # Dockerfile for NGINX
│
└── .github
    └── workflows
        └── ci-cd.yml               # GitHub Actions CI/CD workflow