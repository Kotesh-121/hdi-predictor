# ML - 0027 - Human Development Index (HDI) Predictor

A machine learning web application that predicts a country's Human Development
Index (HDI) score and category from four indicators, using Flask + scikit-learn.

## Folder structure
```
ML - 0027 - Human Development Index/
├── Dataset/
│   └── HDI.csv               # Training dataset (120 countries)
├── Training/
│   └── HumDevIndex.ipynb      # EDA, preprocessing, model training notebook
├── Flask/
│   ├── app.py                 # Flask backend
│   ├── HDI.pkl                 # Serialized trained model
│   ├── templates/
│   │   ├── indexnew.html      # Input form page
│   │   └── resultnew.html     # Prediction result page
│   └── static/
│       └── style.css          # Shared styling
└── README.md
```

## How to run
```bash
cd Flask
pip install flask scikit-learn pandas numpy
python app.py
```
Then open http://127.0.0.1:5000 in your browser.

## Inputs
| Field | Range |
|---|---|
| Country | dropdown (120 countries) |
| Life Expectancy | 30 - 89 years |
| Mean Years of Schooling | 1 - 15 years |
| GNI Per Capita | 740 - 149,000 USD |
| Internet Users | 0 - 100 % |

## Output
- Predicted HDI score (0-1)
- HDI Category: Low (< 0.550) / Medium (0.550-0.699) / High (0.700-0.799) / Very High (>= 0.800)

## Model
Linear Regression (scikit-learn), trained on the four numeric features above.
See `Training/HumDevIndex.ipynb` for the full EDA → preprocessing → training →
evaluation → serialization pipeline (Epics 2-7 of the project workflow).
