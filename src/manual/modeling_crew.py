from crewai import Agent, Task, Crew, Process
from src.tools.code_execution import code_execution_tool


class ManualModelingCrew:
    """
    PATH A: Hardcoded Modeling Crew from the base paper.
    Agents, tasks, and tools are manually defined in Python code.
    This is kept unchanged for comparison with AutoCrew.
    """

    def __init__(self, llm):
        self.llm = llm

    def run(self, dataset_path, task_desc):
        # === AGENTS (hardcoded) ===

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
            max_iter=5,  # writer: no tools, just text
        )

        # === TASKS (hardcoded) ===

        task1 = Task(
            description=(
                f"Use the execute_python_code tool to run this exact Python script:\n"
                f"import pandas as pd\n"
                f"df = pd.read_csv('{dataset_path}')\n"
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
                f"Use the execute_python_code tool to run this EXACT Python script — copy it verbatim:\n\n"
                f"import pandas as pd\n"
                f"import numpy as np\n"
                f"from sklearn.model_selection import train_test_split\n"
                f"from sklearn.ensemble import RandomForestClassifier\n"
                f"from sklearn.preprocessing import LabelEncoder\n"
                f"from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score\n"
                f"df = pd.read_csv('{dataset_path}')\n"
                f"df = df.replace('?', np.nan)\n"
                f"df = df.dropna()\n"
                f"for col in df.select_dtypes(include='object').columns:\n"
                f"    df[col] = LabelEncoder().fit_transform(df[col])\n"
                f"# Target: {task_desc}\n"
                f"known = {{'SeriousDlqin2yrs', 'Class'}}\n"
                f"target_col = next((c for c in known if c in df.columns), df.columns[-1])\n"
                f"X = df.drop(columns=[target_col] + [c for c in df.columns if 'unnamed' in c.lower()])\n"
                f"y = df[target_col]\n"
                f"X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
                f"model = RandomForestClassifier(n_estimators=100, random_state=42)\n"
                f"model.fit(X_train, y_train)\n"
                f"y_pred = model.predict(X_test)\n"
                f"print('Accuracy:', round(accuracy_score(y_test, y_pred), 4))\n"
                f"print('F1 Score:', round(f1_score(y_test, y_pred, average='weighted'), 4))\n"
                f"print('Precision:', round(precision_score(y_test, y_pred, average='weighted'), 4))\n"
                f"print('Recall:', round(recall_score(y_test, y_pred, average='weighted'), 4))\n\n"
                f"Your Final Answer MUST be the exact four lines printed by the code. "
                f"Do not add any other text, JSON, or formatting."
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
                f"- Algorithm: Random Forest (RandomForestClassifier, n_estimators=100, random_state=42)\n"
                f"- Dataset: {dataset_path}\n"
                f"- Task: {task_desc}\n"
                f"- Accuracy: [exact value from previous task]\n"
                f"- F1 Score: [exact value from previous task]\n"
                f"- Precision: [exact value from previous task]\n"
                f"- Recall: [exact value from previous task]\n"
                f"- Preprocessing: replaced '?' with NaN, dropped NaN rows, LabelEncoder on categoricals\n"
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
            verbose=False,
        )

        return crew.kickoff()
