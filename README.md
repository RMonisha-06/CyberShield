\# CyberShield

\### Context-Enhanced Neural Network-Based Cyberbullying Detection with Obfuscated Text Normalization



\## Overview



CyberShield is a machine-learning project designed to detect cyberbullying in online text. It focuses on identifying abusive language, including text deliberately modified using character substitutions, unusual punctuation, repeated characters, and irregular spacing.



The project explores how text normalization and contextual features can improve cyberbullying detection compared with a baseline text-classification approach.



\## Key Features



\- \*\*Cyberbullying Detection:\*\* Classifies text to identify potentially cyberbullying content.

\- \*\*Obfuscated Text Normalization:\*\* Processes modified text patterns such as leetspeak, character deletion, repeated characters, punctuation changes, and unusual spacing.

\- \*\*Context-Enhanced Classification:\*\* Explores contextual features alongside text representations.

\- \*\*Neural Network Models:\*\* Includes baseline and context-enhanced BiLSTM model experiments.

\- \*\*Model Evaluation:\*\* Compares model performance using classification metrics, including precision, recall, and F1-score.

\- \*\*Robustness Evaluation:\*\* Tests model behavior on different categories of obfuscated text.

\- \*\*Web Interface:\*\* Provides an application interface for interacting with the detection system.



\## Problem Statement



Cyberbullying detection systems may struggle when abusive words are intentionally modified to evade conventional text filters. For example, users may replace letters with numbers, insert punctuation, remove characters, or alter spacing.



These modifications can make abusive content harder to identify. CyberShield investigates text normalization and contextual information as complementary approaches to improve detection under such conditions.



\## Proposed Approach



The project follows a machine-learning workflow:



1\. \*\*Dataset Preparation:\*\* Load and inspect datasets, clean the text, and prepare labels.

2\. \*\*Text Preprocessing:\*\* Normalize text and investigate obfuscation patterns.

3\. \*\*Dataset Splitting:\*\* Prepare training, validation, and test datasets.

4\. \*\*Tokenization and Sequence Preparation:\*\* Convert text into numerical representations suitable for neural-network models.

5\. \*\*Context Feature Extraction:\*\* Generate additional features for context-enhanced experiments.

6\. \*\*Model Training:\*\* Train a baseline model and a context-enhanced BiLSTM model.

7\. \*\*Evaluation:\*\* Compare model performance on regular and obfuscated text.

8\. \*\*Application Integration:\*\* Use the application interface to interact with the detection system.



\## Technology Stack



\- \*\*Programming Language:\*\* Python

\- \*\*Machine Learning / Deep Learning:\*\* Neural-network-based text classification

\- \*\*NLP:\*\* Text preprocessing, normalization, tokenization, and sequence preparation

\- \*\*Model Architecture:\*\* Baseline classifier and context-enhanced BiLSTM experiments

\- \*\*Data Processing:\*\* CSV, Excel, NumPy arrays

\- \*\*Web Interface:\*\* HTML templates and Python application

\- \*\*Development Tools:\*\* Visual Studio Code, Git, and GitHub



> The exact libraries and versions are listed in `requirements.txt`.



\## Project Structure



```text

CyberShield/

├── app/

│   └── app.py

├── data/

│   ├── raw/

│   ├── processed/

│   └── obfuscated/

├── extension/

├── models/

│   └── tokenizer.json

├── results/

│   ├── figures/

│   └── metrics/

├── src/

│   ├── normalize\_text.py

│   ├── predict\_text.py

│   ├── prepare\_sequences.py

│   ├── train\_baseline.py

│   ├── train\_context\_bilstm.py

│   └── ...

├── templates/

│   └── index.html

├── fix\_model.py

├── requirements.txt

├── test\_predictions.py

├── .gitignore

└── README.md

```



\## Installation and Setup



\### Prerequisites



\- Python 3.11 or a compatible Python version

\- pip

\- Git



\### 1. Clone the repository



```bash

git clone https://github.com/Monisha-06/CyberShield.git

cd CyberShield

```



Replace the repository URL if your GitHub username or repository URL differs.



\### 2. Create a virtual environment



\*\*Windows PowerShell:\*\*



```powershell

python -m venv venv

.\\venv\\Scripts\\Activate.ps1

```



\### 3. Install dependencies



```powershell

python -m pip install -r requirements.txt

```



\## Running the Application



From the project root, with the virtual environment activated, run:



```powershell

python app\\app.py

```



If the application starts successfully, open the local URL printed in the terminal in your browser.



If a required model file is missing, check the model-loading configuration in `app/app.py` before launching the application.



\## Running Experiments



The `src/` directory contains scripts for data preparation, normalization, model training, and evaluation.



Examples:



```powershell

python src\\normalize\_text.py

python src\\prepare\_sequences.py

python src\\train\_baseline.py

python src\\train\_context\_bilstm.py

```



\*\*Important:\*\* Run these scripts only when you intend to perform the corresponding preprocessing or training step. Check each script's configuration and input requirements first. Some steps may depend on artifacts generated by earlier scripts.



\## Evaluation



The project includes evaluation scripts and result files for analyzing model behavior.



Evaluation areas include:



\- Baseline model performance

\- Context-enhanced model performance

\- Detection under different text-obfuscation transformations

\- Precision, recall, and F1-score comparisons

\- Performance changes after text normalization



The generated figures and metric files are organized under `results/`.



Actual numerical results should be interpreted alongside the relevant dataset, test split, model configuration, and experimental conditions.



\## Intended Applications



CyberShield explores potential applications in:



\- Online community moderation

\- Cyberbullying detection research

\- Social media text analysis

\- Online safety tools

\- Educational research on abusive-language detection



The system is intended as a decision-support tool for identifying potentially harmful text, not as a definitive judgment of a person's intent.



\## Limitations



\- Text classification can produce false positives and false negatives.

\- Sarcasm, friendly teasing, cultural context, and conversational history can affect predictions.

\- Obfuscation normalization may unintentionally change the meaning of text.

\- Performance depends on the quality, coverage, and representativeness of the training data.

\- Predictions should be reviewed in context before any moderation decision is made.



\## Future Improvements



\- Improve contextual understanding of conversations.

\- Evaluate performance across more languages and online communities.

\- Reduce false positives caused by harmless expressions and friendly teasing.

\- Integrate the detection system into a browser extension.

\- Improve inference efficiency and usability.

\- Expand evaluation using additional datasets and independent test sets.



\## Research Focus



CyberShield investigates the combination of text normalization and contextual neural-network features for more robust cyberbullying detection when abusive content is deliberately obfuscated.



\## Responsible Use



This project is intended for research and online-safety applications. Predictions should be interpreted carefully, with appropriate attention to privacy, fairness, and the context in which the text appears.



\## License



A license has not yet been specified. Add an appropriate open-source license before permitting reuse or redistribution.



\## Author



Developed as an academic machine-learning project focused on cyberbullying detection and robust natural-language processing.



