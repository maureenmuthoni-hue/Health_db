#  Healthcare ML Pipeline

An end-to-end machine learning pipeline for healthcare data — covering data ingestion and cleaning, PostgreSQL storage, XGBoost-based prediction, a FastAPI inference endpoint, and automated Airflow retraining DAGs.

---

##  Overview

This project implements a production-style ML pipeline that:

- Ingests and cleans raw healthcare data
- Stores structured data in a PostgreSQL database
- Trains an XGBoost classification model
- Serves predictions via a FastAPI REST endpoint
- Automates periodic model retraining using Apache Airflow DAGs

---

##  Project Structure

```
Healthcare-ml-project/
├── dags/               # Airflow DAGs for automated retraining
├── data/               # Raw and processed datasets
├── models/             # Saved trained model artifacts
├── src/                # Core pipeline source code
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
├── main.py             # FastAPI application entry point
├── requirements.txt    # Python dependencies
├── pyproject.toml      # Project metadata and build config
├── .python-version     # Python version pin
└── README.md
```

---

##  Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| ML Model | XGBoost |
| API Framework | FastAPI |
| Database | PostgreSQL (via psycopg2 + SQLAlchemy) |
| Orchestration | Apache Airflow |
| Data Processing | Pandas, Scikit-learn |
| Deployment | Render |

---

##  Setup & Installation

### Prerequisites

- Python 3.12+
- PostgreSQL running locally or a managed cloud instance
- (Optional) Apache Airflow for DAG scheduling

### 1. Clone the repository

```bash
git clone https://github.com/mainamuragev/Healthcare-ml-project.git
cd Healthcare-ml-project
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/healthcare_db
MODEL_PATH=models/xgboost_model.pkl
```

>  **Important:** Never commit your `.env` file. It is already listed in `.gitignore`.

### 5. Set up the database

Make sure your PostgreSQL instance is running, then initialize the schema:

```bash
python src/data_ingestion.py
```

---

##  Running the API

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

Interactive docs: `http://localhost:8000/docs`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Run inference on patient data |

### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 45, "bmi": 28.5, "glucose": 120, "blood_pressure": 80}'
```

### Example Response

```json
{
  "prediction": 1,
  "probability": 0.83,
  "label": "High Risk"
}
```

---

##  Model Training

To train or retrain the XGBoost model manually:

```bash
python src/train.py
```

The trained model will be saved to the `models/` directory.

---

##  Airflow DAGs

Airflow DAGs in the `dags/` folder handle automated periodic retraining.

### Setting up Airflow

```bash
export AIRFLOW_HOME=$(pwd)/airflow
airflow db init
airflow users create --username admin --password admin \
  --firstname Admin --lastname User --role Admin --email admin@example.com
airflow webserver --port 8080 &
airflow scheduler &
```

Then navigate to `http://localhost:8080` and enable the retraining DAG.

---

##  Deployment on Render

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

>  **Note:** Do **not** run database migrations or model training in the build command. Place them in the start command or a Render release command to avoid build-time connection errors.

### Environment Variables on Render

Set the following in your Render service dashboard:

| Variable | Description |
|---|---|
| `DATABASE_URL` | Full PostgreSQL connection string from your Render DB |
| `MODEL_PATH` | Path to the saved model file |

---

##  Dependencies

Key packages used (see `requirements.txt` for full list):

- `fastapi`, `uvicorn` — API server
- `xgboost`, `scikit-learn` — ML model and preprocessing
- `pandas`, `numpy` — Data manipulation
- `sqlalchemy`, `psycopg2-binary` — Database ORM and driver
- `apache-airflow` — Pipeline orchestration
- `pydantic` — Data validation
- `python-dotenv` — Environment variable management

---

##  License

This project is open source and available under the [MIT License](LICENSE).

---

