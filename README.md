# Citizen Feedback Text Analysis

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete, reproducible data analysis project for analyzing citizen feedback about public services in Nigeria. This project demonstrates end-to-end text analytics including data generation, cleaning, sentiment analysis, topic modeling, and interactive dashboards.

## 📋 Project Overview

This project analyzes citizen complaints, suggestions, and praise about public services (health, education, water, sanitation, local government) to:
- Surface dominant themes through topic modeling
- Track sentiment trends over time and across regions
- Provide actionable recommendations for service improvement
- Enable interactive exploration through a Streamlit dashboard

### Key Features

✅ **Synthetic Data Generation**: Creates realistic feedback data for Nigeria's 37 states  
✅ **Text Processing Pipeline**: Cleaning, PII masking, feature extraction  
✅ **Sentiment Analysis**: Rule-based lexicon approach with negation and intensifier handling  
✅ **Topic Modeling**: LDA and NMF implementations with representative documents  
✅ **Interactive Dashboard**: Streamlit app with filters, visualizations, and data export  
✅ **Comprehensive Documentation**: Data dictionary, ethics guidelines, modeling notes  
✅ **Unit Tests**: pytest-based tests for core pipeline functions  
✅ **Reproducible**: All randomness seeded for consistent results  

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/tiwa-codes/citizen-feedback-text-analysis.git
cd citizen-feedback-text-analysis

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (required for text processing)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt'); nltk.download('omw-1.4')"
```

### Run the Pipeline

**Option 1: Full Pipeline (Automated)**

```bash
python -m src.cli run-pipeline
```

This will:
1. Generate 50,000 synthetic feedback records
2. Clean and preprocess text
3. Compute sentiment scores
4. Run topic modeling (LDA with 10 topics)
5. Generate visualizations

**Option 2: Step-by-Step**

```bash
# 1. Generate synthetic data
python -m src.data.generate_synthetic_feedback --n 50000 --months 24

# 2. Clean text data
python -m src.text.cleaning --input data/raw/citizen_feedback.csv --output data/processed/citizen_feedback_clean.parquet

# 3. Compute sentiment
python -m src.text.sentiment

# 4. Run topic modeling
python -m src.text.topic_modeling --method lda --n-topics 10

# 5. Generate visualizations
python -m src.viz.plots
```

### Launch Dashboard

```bash
streamlit run dashboards/app.py
```

Open your browser to `http://localhost:8501` to explore the interactive dashboard.

### Explore Analysis Notebook

```bash
jupyter lab notebooks/01_citizen_feedback_eda.ipynb
```

---

## 📁 Project Structure

```
citizen-feedback-text-analysis/
├── config/
│   └── analysis_config.yml          # Configuration parameters
├── data/
│   ├── raw/
│   │   └── citizen_feedback.csv     # Generated synthetic data
│   └── processed/
│       ├── citizen_feedback_clean.parquet    # Cleaned data
│       └── topic_assignments.csv             # Topic modeling results
├── notebooks/
│   └── 01_citizen_feedback_eda.ipynb         # Exploratory analysis notebook
├── src/
│   ├── data/
│   │   ├── generate_synthetic_feedback.py    # Data generation script
│   │   └── sampling_helpers.py               # Sampling utilities
│   ├── text/
│   │   ├── cleaning.py                       # Text cleaning & PII masking
│   │   ├── features.py                       # TF-IDF feature extraction
│   │   ├── sentiment.py                      # Sentiment analysis
│   │   └── topic_modeling.py                 # LDA/NMF topic modeling
│   ├── viz/
│   │   └── plots.py                          # Visualization functions
│   └── cli.py                                # Command-line interface
├── dashboards/
│   └── app.py                                # Streamlit dashboard
├── docs/
│   ├── data_dictionary.md                    # Dataset field descriptions
│   ├── ethics_guidelines.md                  # Ethics & privacy guidance
│   └── modeling_notes.md                     # Technical methodology
├── reports/
│   ├── citizen_feedback_brief.md             # Policy brief (1-2 pages)
│   └── figures/                              # Generated plots
├── tests/
│   ├── test_cleaning.py                      # Unit tests for cleaning
│   ├── test_sentiment.py                     # Unit tests for sentiment
│   └── test_topic_modeling.py                # Unit tests for topics
├── requirements.txt                          # Python dependencies
├── .gitignore                                # Git ignore patterns
└── README.md                                 # This file
```

---

## 🔧 Configuration

Edit `config/analysis_config.yml` to customize:

```yaml
random_seed: 42                 # Reproducibility
n_topics: 10                    # Number of topics for modeling
min_df: 5                       # Min document frequency for TF-IDF
max_df: 0.95                    # Max document frequency for TF-IDF
n_features_tfidf: 20000         # Max TF-IDF features
```

---

## 📊 Data

### Synthetic Data

This project uses **synthetic data** to demonstrate the analysis pipeline without privacy concerns. The generated dataset includes:

- **50,000 feedback records** spanning 24 months (2024-2025)
- **37 states** of Nigeria with realistic LGAs
- **5 channels**: SMS, Hotline, Web Form, In-person, Social Media
- **Realistic text patterns**: Nigerian English expressions, typos, abbreviations
- **Quality issues**: ~3-5% duplicates, ~2% spam (by design)

### Fields

| Field | Description |
|-------|-------------|
| feedback_id | Unique identifier (FB000001, ...) |
| created_at | Submission date (YYYY-MM-DD) |
| state | Nigerian state |
| lga | Local Government Area |
| channel | Submission channel |
| facility_or_service | Facility or service mentioned |
| raw_text | Original feedback message |
| rating | Optional 1-5 rating |
| response_time_days | Days to first response (if resolved) |
| resolved | True/False |
| assigned_dept | Health, Education, Water, etc. |

See `docs/data_dictionary.md` for complete field descriptions.

---

## 🔍 Analysis Components

### 1. Text Cleaning (`src/text/cleaning.py`)

- Remove non-printable characters
- Mask PII (phone numbers, emails, IDs) → `[PHONE]`, `[EMAIL]`, `[ID]`
- Expand contractions ("dont" → "do not")
- Normalize punctuation
- Calculate metadata (word count, character count, etc.)

### 2. Sentiment Analysis (`src/text/sentiment.py`)

- **Rule-based lexicon** with ~90 positive/negative words
- **VADER-like heuristics**: negation handling, intensifiers
- **Output**: sentiment_score (-1 to 1), sentiment_label (positive/neutral/negative)
- **Accuracy**: ~75-80% on manual review

### 3. Topic Modeling (`src/text/topic_modeling.py`)

- **LDA** (Latent Dirichlet Allocation) or **NMF** (Non-negative Matrix Factorization)
- **Preprocessing**: stopword removal, lemmatization, tokenization
- **Output**: 10 topics with top terms and representative documents
- **Coherence**: Simple proxy metric provided

### 4. Visualizations (`src/viz/plots.py`)

- Topic trends over time (area chart)
- Sentiment trends (line chart)
- Top keywords (bar chart)
- Topic × Sentiment heatmap
- Channel and state distributions

### 5. Dashboard (`dashboards/app.py`)

- **Filters**: State, date range, channel, department, topic, sentiment
- **KPIs**: Total messages, % negative, % unresolved, avg response time
- **Interactive plots**: Plotly charts with hover details
- **Data export**: Download filtered data as CSV

---

## 🧪 Testing

Run unit tests with pytest:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_cleaning.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

Tests cover:
- Text cleaning and PII masking
- Sentiment analysis scoring
- Topic modeling pipeline
- Edge cases (empty text, None values, short messages)

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [data_dictionary.md](docs/data_dictionary.md) | Complete field descriptions, data types, allowed values |
| [ethics_guidelines.md](docs/ethics_guidelines.md) | Privacy, bias, responsible interpretation |
| [modeling_notes.md](docs/modeling_notes.md) | Technical methodology, parameters, limitations |
| [citizen_feedback_brief.md](reports/citizen_feedback_brief.md) | Policy brief with findings and recommendations |

---

## ⚠️ Ethics & Privacy

This project demonstrates **privacy-preserving practices** even on synthetic data:

- **PII Masking**: Phone numbers, emails, IDs are masked
- **Aggregation**: Results reported at group level (state, LGA, channel)
- **Transparency**: Clear documentation of data sources and limitations
- **Bias Awareness**: Known biases (selection, language, algorithmic) are documented

When adapting for **real data**:
1. Conduct privacy impact assessment
2. Obtain necessary consents and approvals
3. Comply with data protection laws (e.g., NDPR in Nigeria)
4. Involve diverse stakeholders in interpretation

See `docs/ethics_guidelines.md` for detailed guidance.

---

## 🛠️ Technical Stack

- **Python 3.11**
- **Data Processing**: pandas, numpy, scipy
- **NLP**: NLTK, scikit-learn, gensim
- **Visualization**: matplotlib, seaborn, plotly
- **Dashboard**: Streamlit
- **Storage**: Parquet (processed data), CSV (raw data)
- **Testing**: pytest

---

## 🔄 Reproducibility

All randomness is seeded with `random_seed: 42`:
- Data generation produces identical records
- Topic modeling produces identical topic assignments
- Results are fully reproducible across runs

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 👥 Authors

- **Project**: Citizen Feedback Text Analysis
- **Purpose**: Educational and demonstration
- **Data**: Synthetic (not real citizen feedback)

---

## 🙏 Acknowledgments

- Nigerian states data sourced from official government records
- Inspired by real-world citizen engagement platforms
- Built with open-source tools and libraries

---

## 📧 Contact

For questions, issues, or collaboration opportunities:
- Open an issue on GitHub
- Review documentation in `docs/` folder
- Check FAQ in wiki (coming soon)

---

## 🎯 Use Cases

This project can be adapted for:
- **Government agencies**: Analyze real citizen feedback systems
- **NGOs**: Monitor service delivery and community needs
- **Researchers**: Study text analytics and NLP on public sector data
- **Students**: Learn text mining, topic modeling, and dashboard development
- **Data scientists**: Template for end-to-end NLP projects

---

## ⚡ Performance

- **Data generation**: ~2 minutes for 50k records
- **Text cleaning**: ~1 minute for 50k records
- **Topic modeling**: ~5-7 minutes for 50k records (LDA, 10 topics)
- **Dashboard**: Loads in <3 seconds with cached data

Tested on: 4-core CPU, 16GB RAM

---

## 🚧 Future Enhancements

- [ ] Multilingual support (Hausa, Yoruba, Igbo, Pidgin)
- [ ] Named Entity Recognition for facilities and locations
- [ ] Aspect-based sentiment (sentiment per topic)
- [ ] Dynamic topic modeling (track topic evolution)
- [ ] Predictive modeling (response time, urgency classification)
- [ ] Integration with real-time data streams (Twitter API, SMS gateway)

---

**Last Updated**: November 2025  
**Version**: 1.0.0