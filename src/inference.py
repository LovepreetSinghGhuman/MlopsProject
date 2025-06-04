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
│   │   └── service.yaml            # Kubernetes service configuration
│   └── /nginx
│       ├── nginx.conf              # NGINX configuration
│       └── Dockerfile              # Dockerfile for NGINX
│
├── /frontend
│   ├── /src
│   │   ├── App.js                  # Main React component
│   │   ├── index.js                # Entry point for React app
│   │   └── package.json             # Frontend dependencies
│   └── Dockerfile                   # Dockerfile for React app
│
├── /ci-cd
│   ├── .github
│   │   └── workflows
│   │       └── ci-cd.yml           # GitHub Actions CI/CD pipeline
│
└── README.md                       # Project documentation