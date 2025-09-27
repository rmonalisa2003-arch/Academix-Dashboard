# Academix-Dashboard
# 🎓 Academix – Where Data Meets Education

Academix is an interactive **student performance dashboard** with **ML-based score prediction**. Visualize performance, explore trends, and predict Math, Reading, and Writing scores for students.  

---

## Features

- **Interactive Filters:** Gender, Ethnic Group, Parent Education.
- **Visualizations:** Countplots, heatmaps, and boxplots for insights.
- **Multi-output ML Prediction:** Predict scores and view RMSE.
- **Train & Retrain:** Train on full or filtered dataset.
- **User-friendly UI:** Built with Streamlit.

---

## Installation

1. Clone the repo:
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

2. Install dependencies:
pip install -r requirements.txt

3. Run dashboard:
streamlit run dashboard.py

## Dataset
Place Student_scores.csv in the project folder. Ensure it has:
MathScore, ReadingScore, WritingScore, Gender, EthnicGroup, ParentEduc, ParentMaritalStatus, WklyStudyHours

## Motivational Note

"Every student has the potential to shine; guidance from teachers and support from parents can turn data into growth."

## Tech Stack

1. Python, Streamlit, Pandas, NumPy
2. Matplotlib, Seaborn
3. Scikit-learn / LightGBM
4. Joblib

