/BERTopic-Azure-ML
│
├── /backend
│   ├── /app
│   │   ├── main.py               # FastAPI application
│   │   ├── model.py              # Model training and prediction logic
│   │   └── requirements.txt       # Python dependencies
│   ├── Dockerfile                 # Dockerfile for FastAPI
│   └── azureml_config.json        # Azure ML configuration
│
├── /frontend
│   ├── index.html                 # Basic HTML frontend
│   └── styles.css                 # CSS for styling
│
├── /nginx
│   ├── nginx.conf                 # NGINX configuration
│   └── Dockerfile                 # Dockerfile for NGINX
│
├── .github
│   └── workflows
│       └── ci-cd.yml             # GitHub Actions CI/CD workflow
│
└── README.md                      # Project documentation