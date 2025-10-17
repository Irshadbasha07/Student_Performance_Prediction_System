# Predicting Student Performance Using Machine Learning

**Project:** final_year_project  
**Description:** A small end-to-end demo that trains a regression model to predict a student's present year percentage based on features like parental education, internet access, study hours, attendance, and previous year percentage. The project includes a Flask web app to serve predictions.

## Structure

```
final_year_project/
├── README.md
├── requirements.txt
├── data/
│   └── student_performance.csv
├── notebooks/
│   └── EDA_and_Model_Training.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── app.py
└── saved_models/
    └── student_model.pkl
```

## Setup (local)

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Retrain model:
   ```bash
   python src/model_training.py
   ```

4. Run Flask app:
   ```bash
   export FLASK_APP=src/app.py
   flask run
   ```
   Or on Windows PowerShell:
   ```powershell
   set FLASK_APP=src/app.py
   flask run
   ```

Open http://127.0.0.1:5000 in your browser and use the web form to predict.

## Notes
- The dataset provided is synthetic and small — it's for demo purposes only.
- For production, use more data, proper preprocessing, cross-validation, and model versioning.irshad
