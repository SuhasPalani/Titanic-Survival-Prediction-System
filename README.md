# 🚢 Titanic Survival Prediction System

A comprehensive machine learning system that predicts passenger survival on the RMS Titanic with real-time streaming capabilities, advanced filtering options, and a modern web interface.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![Kafka](https://img.shields.io/badge/Apache%20Kafka-Latest-red.svg)](https://kafka.apache.org/)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

- **🤖 Machine Learning Models**: Multiple algorithms with hyperparameter tuning (Random Forest, Logistic Regression, SVM)
- **🧹 Data Processing**: Comprehensive cleaning and feature engineering pipeline
- **📊 Exploratory Analysis**: Rich visualizations and statistical insights
- **⚡ Real-time Predictions**: Kafka streaming for live prediction simulation
- **🔍 Advanced Filtering**: Filter by class, gender, age range, survival prediction
- **🌐 Web Interface**: Modern, responsive UI for easy predictions
- **🔌 REST API**: Complete API with batch processing capabilities
- **🐳 Docker Support**: Containerized deployment with Docker Compose
- **🧪 Testing Suite**: Comprehensive unit tests for all components



### Real-time Streaming
```bash
Passenger 2708: SURVIVED (Probability: 0.823)
Passenger 9909: DID NOT SURVIVE (Probability: 0.234)
--- STATISTICS ---
Processed: 50 passengers
Survival Rate: 0.640
Avg Survival Probability: 0.567
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker & Docker Compose (for full features)
- Git

### 1. Clone & Setup
```bash
git clone https://github.com/your-username/titanic-survival-prediction.git
cd titanic-survival-prediction
pip install -r requirements.txt
```

### 2. Download Data & Train Model
```bash
# Download Titanic dataset
python scripts/download_data.py

# Train the machine learning model
python scripts/train_model.py
```

### 3. Start Web Interface
```bash
# Simple approach - just the web interface
python api/app.py
```
Visit `http://localhost:5000` 🎉

### 4. Full System with Kafka (Optional)
```bash
# Start all services (Kafka + API)
docker-compose up -d

# Run real-time simulation
python scripts/run_simulation.py
```



## 🎯 Usage Examples

### Web Interface
1. Open `http://localhost:5000`
2. Fill passenger details:
   - **Class**: 1st, 2nd, or 3rd
   - **Gender**: Male or Female  
   - **Age**: 0-120 years
   - **Family**: Siblings/Spouses and Parents/Children aboard
   - **Fare**: Ticket price paid
   - **Port**: Embarkation port (Cherbourg, Queenstown, Southampton)
3. Click "🔮 Predict Survival"
4. View results with survival probability

### API Endpoints

#### Single Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pclass": 1,
    "Sex": "female",
    "Age": 25,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 100.0,
    "Embarked": "S"
  }'
```

**Response:**
```json
{
  "prediction": 1,
  "survival_probability": 0.89,
  "death_probability": 0.11,
  "status": "Survived"
}
```

#### Batch Prediction with Filters
```bash
curl -X POST http://localhost:5000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "passengers": [
      {"Pclass": 1, "Sex": "female", "Age": 25, "SibSp": 0, "Parch": 0, "Fare": 100, "Embarked": "S"},
      {"Pclass": 3, "Sex": "male", "Age": 30, "SibSp": 1, "Parch": 0, "Fare": 15, "Embarked": "S"}
    ],
    "filters": {
      "pclass": [1, 2],
      "sex": "female"
    },
    "sort_by": "survival_probability",
    "ascending": false
  }'
```

### Real-time Streaming
```bash
# Start Kafka services
docker-compose up -d

# Run simulation (generates synthetic passengers)
python scripts/run_simulation.py

# Watch live predictions stream by
```

## 🧠 Machine Learning Pipeline

### Data Processing
- **Missing Value Handling**: Age filled by median per class/gender, Embarked by mode
- **Feature Engineering**: 
  - `FamilySize` = SibSp + Parch + 1
  - `IsAlone` = 1 if FamilySize == 1, else 0
  - `Title` = Extracted from Name (Mr, Mrs, Miss, Master, Rare)
  - `AgeGroup` = Categorical age ranges
  - `FareGroup` = Quartile-based fare categories
- **Encoding**: One-hot encoding for categorical variables

### Models & Performance
| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-----------|----------|-----------|---------|----------|---------|
| Random Forest | 84.2% | 81.5% | 78.9% | 80.2% | 87.1% |
| Logistic Regression | 82.8% | 79.3% | 77.2% | 78.2% | 85.6% |
| SVM | 83.1% | 80.1% | 76.8% | 78.4% | 86.2% |

*System automatically selects best performing model*

### Feature Importance (Random Forest)
1. **Sex** (0.28) - Gender was the strongest survival predictor
2. **Fare** (0.26) - Higher fare = better survival chances  
3. **Age** (0.18) - "Women and children first" policy
4. **Title** (0.12) - Social status indicator
5. **Pclass** (0.09) - Passenger class affected access to lifeboats
6. **FamilySize** (0.04) - Small families had better chances
7. **Embarked** (0.03) - Port of embarkation had minor impact

## 🔍 Advanced Features

### Filtering & Sorting Options
- **Filter by**: Passenger class, gender, age range, survival prediction
- **Sort by**: Survival probability, age, fare, class, any feature
- **Custom queries**: Combine multiple filters for targeted analysis

### Real-time Analytics
- Live survival rate tracking
- Processing statistics
- Throughput monitoring
- Error rate analysis

### Data Insights Discovered
- **Women** had 74% survival rate vs **men** at 19%
- **1st class** passengers: 63% survival vs **3rd class**: 24%
- **Children under 10**: 61% survival rate
- **Optimal family size**: 2-4 members had highest survival
- **Port matters**: Cherbourg passengers had higher survival (55%)

## 🐳 Docker Deployment

### Development Setup
```bash
# Build and start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f titanic-api

# Stop services
docker-compose down
```

### Production Deployment
```bash
# Build production image
docker build -t titanic-predictor:latest .

# Run container
docker run -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  titanic-predictor:latest
```

## 🧪 Testing

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Test Categories
```bash
# Data processing tests
python -m pytest tests/test_data_processing.py -v

# Model tests  
python -m pytest tests/test_models.py -v

# Kafka streaming tests
python -m pytest tests/test_kafka.py -v
```

### Test Coverage
- Data cleaning and feature engineering
- Model training and evaluation  
- Prediction accuracy and filtering
- Kafka producer/consumer functionality
- API endpoints and error handling

## 📊 Jupyter Notebooks

### Data Exploration
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```
- Comprehensive EDA with visualizations
- Statistical analysis of survival patterns
- Feature correlation analysis
- Data quality assessment

### Model Development
```bash
jupyter notebook notebooks/02_model_development.ipynb  
```
- Model comparison and selection
- Hyperparameter tuning process
- Feature importance analysis
- Performance evaluation

## 🛠️ Configuration

### Environment Variables
```bash
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
export KAFKA_TOPIC=titanic-predictions
export API_HOST=0.0.0.0
export API_PORT=5000
export DEBUG=True
```

### Configuration File
Edit `config/config.py` to customize:
- File paths and directories
- Kafka connection settings  
- API configuration
- Model parameters
- Logging settings

## 🚨 Troubleshooting

### Common Issues

#### Model Not Found
```bash
# Error: FileNotFoundError: models/saved_models/titanic_model.pkl
# Solution: Train the model first
python scripts/train_model.py
```

#### Kafka Connection Failed  
```bash
# Error: NoBrokersAvailable
# Solution: Start Kafka services
docker-compose up -d kafka zookeeper
```

#### Port Already in Use
```bash
# Error: Address already in use: 5000
# Solution: Change port in config or kill process
lsof -ti:5000 | xargs kill -9
# OR change port in config/config.py
```

#### Missing Data Files
```bash
# Error: FileNotFoundError: data/raw/train.csv
# Solution: Download dataset
python scripts/download_data.py
```
