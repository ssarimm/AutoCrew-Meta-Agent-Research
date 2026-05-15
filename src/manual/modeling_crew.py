from crewai import Agent, Task, Crew, Process
from src.tools.code_execution import code_execution_tool


def _build_loading_code(dataset_path, task_desc):
    """
    Build dataset-specific loading/preprocessing code snippet.
    Returns (load_code, target_detection_code, rf_extra_params).
    """
    if "credit_card_approval" in dataset_path:
        load_code = (
            f"df = pd.read_csv('{dataset_path}', header=None)\n"
            f"df = df.replace('?', np.nan)\n"
            f"df = df.dropna()\n"
            f"for col in df.select_dtypes(include='object').columns:\n"
            f"    df[col] = LabelEncoder().fit_transform(df[col])\n"
        )
        target_code = (
            f"target_col = df.columns[-1]\n"
            f"X = df.drop(columns=[target_col])\n"
            f"y = df[target_col]\n"
        )
        rf_params = "n_estimators=100, random_state=42"

    elif "cs-training" in dataset_path:
        load_code = (
            f"df = pd.read_csv('{dataset_path}')\n"
            f"df = df.drop(columns=[c for c in df.columns if 'unnamed' in c.lower()], errors='ignore')\n"
            f"df = df.fillna(df.median(numeric_only=True))\n"
            f"for col in df.select_dtypes(include='object').columns:\n"
            f"    df[col] = LabelEncoder().fit_transform(df[col])\n"
        )
        target_code = (
            f"target_col = 'SeriousDlqin2yrs'\n"
            f"X = df.drop(columns=[target_col])\n"
            f"y = df[target_col]\n"
        )
        rf_params = "n_estimators=100, random_state=42"

    elif "creditcard_2023" in dataset_path:
        load_code = (
            f"df = pd.read_csv('{dataset_path}')\n"
            f"df = df.drop(columns=[c for c in df.columns if c.lower() in ('id', 'unnamed: 0')], errors='ignore')\n"
            f"df = df.dropna()\n"
            f"for col in df.select_dtypes(include='object').columns:\n"
            f"    df[col] = LabelEncoder().fit_transform(df[col])\n"
        )
        target_code = (
            f"target_col = 'Class'\n"
            f"X = df.drop(columns=[target_col])\n"
            f"y = df[target_col]\n"
        )
        # Large dataset: fewer trees, parallel, depth-limited to avoid timeout
        rf_params = "n_estimators=20, max_depth=15, n_jobs=-1, random_state=42, class_weight='balanced'"

    else:
        # Generic fallback
        load_code = (
            f"df = pd.read_csv('{dataset_path}')\n"
            f"df = df.replace('?', np.nan)\n"
            f"df = df.dropna()\n"
            f"for col in df.select_dtypes(include='object').columns:\n"
            f"    df[col] = LabelEncoder().fit_transform(df[col])\n"
        )
        target_code = (
            f"known = {{'SeriousDlqin2yrs', 'Class'}}\n"
            f"target_col = next((c for c in known if c in df.columns), df.columns[-1])\n"
            f"X = df.drop(columns=[target_col] + [c for c in df.columns if 'unnamed' in c.lower()])\n"
            f"y = df[target_col]\n"
        )
        rf_params = "n_estimators=100, random_state=42"

    return load_code, target_code, rf_params


def _get_sampling_code(dataset_path):
    """Return optional training data sampling code for large datasets."""
    if "creditcard_2023" in dataset_path:
        return (
            f"# Sample training data if too large (keeps all fraud cases via stratify)\n"
            f"if len(X_train) > 100000:\n"
            f"    from sklearn.utils import resample\n"
            f"    import pandas as _pd_tmp\n"
            f"    _combined = _pd_tmp.concat([X_train, y_train], axis=1)\n"
            f"    _majority = _combined[_combined[target_col] == 0]\n"
            f"    _minority = _combined[_combined[target_col] == 1]\n"
            f"    _majority_down = _majority.sample(n=min(50000, len(_majority)), random_state=42)\n"
            f"    _combined = _pd_tmp.concat([_majority_down, _minority])\n"
            f"    X_train = _combined.drop(columns=[target_col])\n"
            f"    y_train = _combined[target_col]\n"
            f"    print('Downsampled training set:', X_train.shape)\n"
        )
    return ""


class ManualModelingCrew:
    """
    PATH A: Hardcoded Modeling Crew from the base paper.
    """

    def __init__(self, llm):
        self.llm = llm

    def run(self, dataset_path, task_desc):
        load_code, target_code, rf_params = _build_loading_code(dataset_path, task_desc)
        sampling_code = _get_sampling_code(dataset_path)

        # === AGENTS ===

        analyst = Agent(
            role="Data Analyst",
            goal="Load data and check for issues.",
            backstory="You allow the engineer to do their job by ensuring data is loadable.",
            tools=[code_execution_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=8,
        )

        engineer = Agent(
            role="ML Engineer",
            goal="Train a model and REPORT METRICS.",
            backstory="You write code to train models. You MUST print the Accuracy and F1 Score.",
            tools=[code_execution_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=8,
        )

        writer = Agent(
            role="Technical Writer",
            goal="Summarize the technical findings.",
            backstory="You read the output from the engineer and write a Model Card.",
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=5,
        )

        # === TASKS ===

        header_param = ", header=None" if "credit_card_approval" in dataset_path else ""

        task1 = Task(
            description=(
                f"Use the execute_python_code tool to run this exact Python script:\n"
                f"import pandas as pd\n"
                f"import numpy as np\n"
                f"from sklearn.preprocessing import LabelEncoder\n"
                f"df = pd.read_csv('{dataset_path}'{header_param})\n"
                f"df = df.replace('?', np.nan)\n"
                f"df = df.dropna()\n"
                f"for col in df.select_dtypes(include='object').columns:\n"
                f"    df[col] = LabelEncoder().fit_transform(df[col])\n"
                f"print('Shape:', df.shape)\n"
                f"print('Columns:', list(df.columns))\n"
                f"print('Missing values:', df.isnull().sum().sum())\n"
                f"print(df.dtypes)"
            ),
            expected_output="Dataset shape, column names, dtypes, and missing value count.",
            agent=analyst,
        )

        task2 = Task(
            description=(
                f"Use the execute_python_code tool to run this exact Python script:\n"
                f"import pandas as pd\n"
                f"import numpy as np\n"
                f"from sklearn.model_selection import train_test_split\n"
                f"from sklearn.ensemble import RandomForestClassifier\n"
                f"from sklearn.preprocessing import LabelEncoder\n"
                f"from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score\n"
                f"{load_code}"
                f"# Target: {task_desc}\n"
                f"{target_code}"
                f"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
                f"{sampling_code}"
                f"model = RandomForestClassifier({rf_params})\n"
                f"model.fit(X_train, y_train)\n"
                f"y_pred = model.predict(X_test)\n"
                f"print('Accuracy:', round(accuracy_score(y_test, y_pred), 4))\n"
                f"print('F1 Score:', round(f1_score(y_test, y_pred, average='weighted'), 4))\n"
                f"print('Precision:', round(precision_score(y_test, y_pred, average='weighted'), 4))\n"
                f"print('Recall:', round(recall_score(y_test, y_pred, average='weighted'), 4))"
            ),
            expected_output=(
                "Exactly four lines:\n"
                "Accuracy: X.XXXX\n"
                "F1 Score: X.XXXX\n"
                "Precision: X.XXXX\n"
                "Recall: X.XXXX"
            ),
            agent=engineer,
            context=[task1],
        )

        task3 = Task(
            description=(
                f"Write a Model Card based on the metrics from the previous task. "
                f"You MUST include ALL of the following fields explicitly:\n"
                f"- Model Name: Random Forest Classifier\n"
                f"- Algorithm: Random Forest (RandomForestClassifier, {rf_params})\n"
                f"- Dataset: {dataset_path}\n"
                f"- Task: {task_desc}\n"
                f"- Accuracy: [exact value from previous task]\n"
                f"- F1 Score: [exact value from previous task]\n"
                f"- Precision: [exact value from previous task]\n"
                f"- Recall: [exact value from previous task]\n"
                f"- Preprocessing: dataset-specific cleaning, LabelEncoder on categoricals\n"
                f"- Limitations: Tree-based model, may not generalize to out-of-distribution data\n"
                f"- Intended Use: Financial risk classification\n"
                f"State every field on a separate line. Do NOT omit any field."
            ),
            expected_output=(
                "Model Card with: Model Name, Algorithm, Dataset, Task, "
                "Accuracy, F1 Score, Precision, Recall, Preprocessing, Limitations, Intended Use."
            ),
            agent=writer,
            context=[task2],
        )

        # === CREW ===

        crew = Crew(
            agents=[analyst, engineer, writer],
            tasks=[task1, task2, task3],
            process=Process.sequential,
            memory=False,
            verbose=True,
            tracing=False,
            max_rpm=5
        )

        return crew.kickoff()
