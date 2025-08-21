## 🚀 Quick Start

To get a preview of our system running:

1.  **Clone this repo** and navigate to its directory.
2.  **Install dependencies:** 
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the exploratory data analysis** to see our initial findings:
    ```bash
    jupyter notebook notebooks/01_EDA.ipynb
    ```
4.  (Optional)**Simulate our training pipeline** (This will run with dummy data to show the structure):
    ```bash
    python src/train.py
    ```
5.  **Launch the demo dashboard:** 
    ```bash
    streamlit run app.py
    ```

*Note: The full pipeline is designed to run on the hackathon datasets, which will be placed in the `data/` directory upon arrival.*

## 🧠 Our Technical Approach

Our solution is built as a modular, end-to-end pipeline:

1.  **Data Fusion:** We merge router logs, topology, config history, and external events on `device_name` and `timestamp`.
2.  **Feature Engineering:** We create temporal features (`hour_of_day`, `is_weekend`), critical ratios (`bandwidth_utilization`), and **rolling time-series features** (e.g., `utilization_rolling_avg_1h`) to give our model temporal "memory".
3.  **Modeling:** We use **XGBoost** for its superior performance on tabular data and use **time-series cross-validation** to avoid leakage and ensure robustness.
4.  **Interpretability:** We integrate **SHAP** to explain predictions, building trust with network engineers.
5.  **Actionable Output:** Our recommendation engine uses a **"Robin Hood" algorithm** to safely reallocate bandwidth from under-utilized nodes without causing secondary congestion.
6.  **Live Demo:** We deliver insights through a **Streamlit dashboard** for real-time monitoring.

## ⏭️ Next Steps at the Hackathon

With the provided data, we will immediately:

1.  Execute the full `data_pipeline.py` on the real datasets.
2.  Perform hyperparameter tuning on our XGBoost model using `train.py`.
3.  Finalize and test the recommendation logic in `recommend.py`.
4.  Build out the production-grade Streamlit dashboard in `app.py`.

## 👥 Team
iNETWORK

Members:
- Aryan
- Ayush(lead)
- Priyanshu
- Piyush

---
*Built for the Genesis Hackathon*


