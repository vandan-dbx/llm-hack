# Databricks notebook source
# MAGIC %pip install -U -qqqq databricks-agents mlflow mlflow-skinny databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import pandas as pd
from databricks import agents

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

# MAGIC %md
# MAGIC # Load your evaluation set from the previous step

# COMMAND ----------

# df = spark.table(EVALUATION_SET_FQN)
# eval_df = df.toPandas()
# display(eval_df)

# COMMAND ----------

# If you did not collect feedback from your stakeholders, and want to evaluate using a manually curated set of questions, you can use the structure below.

eval_data = [
    {
        ### REQUIRED
        # Question that is asked by the user
        "request": "How does the epitranscriptome influence the progression of atherosclerosis?",

        ### OPTIONAL
        # Optional, user specified to identify each row
        "request_id": "test_id_1",
        # Optional: correct response to the question
        # If provided, Agent Evaluation can compute additional metrics.
        "expected_response": "The epitranscriptome, through modifications such as m6A, m5C, and A-to-I editing, plays a critical regulatory role in atherosclerosis by affecting RNA molecules like mRNA and ncRNAs. These modifications alter gene expression and cellular pathways involved in disease progression. Targeting RNA-modifying enzymes offers potential therapeutic opportunities for treating atherosclerosis.",
        # Optional: Which documents should be retrieved.
        # If provided, Agent Evaluation can compute additional metrics.
        "expected_retrieved_context": [
            {
                # URI of the relevant document to answer the request
                # Must match the contents of `document_uri` in your chain config / Vec
                "doc_uri": "Epitranscriptome: A Novel Regulatory Layer during Atherosclerosis Progression",
            },
        ],
    },
        {
        ### REQUIRED
        # Question that is asked by the user
        "request": " What is the basic principle behind the lateral transshipment policy used in managing short-term vaccine inventories?",

        ### OPTIONAL
        # Optional, user specified to identify each row
        "request_id": "test_id_2",
        # Optional: correct response to the question
        # If provided, Agent Evaluation can compute additional metrics.
        "expected_response": """The lateral transshipment policy, as applied in managing short-term vaccine inventories, is detailed further by utilizing a model based on availability to decide the quantity of vaccines to transfer—specifically, the amount is determined to be less than or equal to a vial’s worth between facilities. This policy is integrated with a reordering system that follows the (s, S) inventory strategy, where s and S represent the reorder point and the maximum desired inventory level, respectively.

In practice, this transshipment approach proves particularly advantageous in stochastic environments where demand for vaccines fluctuates unpredictably across different locations. For instance, if one hospital experiences lower-than-expected demand and has excess vaccines, while another faces higher demand, the lateral transshipment policy allows the surplus vaccines to be redirected where they are needed most. This not only ensures more efficient use of the vaccine supply, reducing instances of vaccine expiration and associated losses, but also helps in maintaining an optimal level of stock across facilities.

This dynamic adjustment of inventory levels is crucial in scenarios where vaccine supply and demand can vary significantly due to factors like logistical challenges, population density variations, or sudden outbreaks of disease. By implementing this policy, hospitals can maintain better vaccine availability, reduce costs associated with overages and shortages, and enhance overall vaccination efforts, which is critical in managing public health effectively.

Moreover, the use of simulations and models, like MILP (Mixed Integer Linear Programming) and DES (Discrete Event Simulation), helps in fine-tuning the policy by predicting outcomes under various scenarios, thereby aiding in the decision-making process and ensuring the policy's effectiveness over different operational conditions. This methodical approach enables healthcare facilities to adapt quickly to changing conditions, making the lateral transshipment policy a vital tool in vaccine inventory management.""",
        # Optional: Which documents should be retrieved.
        # If provided, Agent Evaluation can compute additional metrics.
        "expected_retrieved_context": [
            {
                # URI of the relevant document to answer the request
                # Must match the contents of `document_uri` in your chain config / Vec
                "doc_uri": "Analyzing economic effect on mRNA vaccine inventory management with redistribution policy.",
            },
        ],
    },
     {
        ### REQUIRED
        # Question that is asked by the user
        "request": "How does hsa_circ_0007482 correlate with pterygium severity and what potential does it hold for managing the condition?",

        ### OPTIONAL
        # Optional, user specified to identify each row
        "request_id": "test_id_3",
        # Optional: correct response to the question
        # If provided, Agent Evaluation can compute additional metrics.
        "expected_response": "Hsa_circ_0007482 is significantly upregulated in pterygium tissues, with its levels positively associated with the severity, vascular density, and thickness of pterygium. Silencing this circRNA inhibits the proliferation of human pterygium fibroblasts (HPFs) and promotes their apoptosis. This suggests that hsa_circ_0007482 plays a critical role in pterygium biology and could serve as a valuable biomarker and a potential non-surgical treatment target for managing the condition.",
        # Optional: Which documents should be retrieved.
        # If provided, Agent Evaluation can compute additional metrics.
        "expected_retrieved_context": [
            {
                # URI of the relevant document to answer the request
                # Must match the contents of `document_uri` in your chain config / Vec
                "doc_uri": "Role of hsa_circ_0007482 in pterygium development: insights into proliferation, apoptosis, and clinical correlations.",
            },
        ],
    },
          {
        ### REQUIRED
        # Question that is asked by the user
        "request": "What challenges are associated with developing targeted drugs against blebbing in tumor metastasis, and how do the agents Blebbistatin, BTS, and BDM compare in this context?",

        ### OPTIONAL
        # Optional, user specified to identify each row
        "request_id": "test_id_4",
        # Optional: correct response to the question
        # If provided, Agent Evaluation can compute additional metrics.
        "expected_response": """Developing targeted drugs against blebbing in tumor metastasis faces challenges due to the limited understanding of the underlying regulatory mechanisms. Among the available agents, Blebbistatin and BTS are effective in suppressing cell blebbing and may help mitigate tumor cell invasion and metastasis. However, their long-term safety and clinical applicability require further rigorous evaluation. In contrast, BDM shows modest efficacy in inhibiting blebbing and demonstrates cytotoxicity in certain contexts, necessitating more comprehensive investigation into its safety and translational potential. Thus, selecting appropriate therapeutic interventions for tumor cell blebbing requires careful consideration of each agent's efficacy, safety, and feasibility for clinical use.""",
        # Optional: Which documents should be retrieved.
        # If provided, Agent Evaluation can compute additional metrics.
        "expected_retrieved_context": [
            {
                # URI of the relevant document to answer the request
                # Must match the contents of `document_uri` in your chain config / Vec
                "doc_uri": "Effects of Corticosteroid Treatment on Olfactory Dysfunction in LATY136F Knock-In Mice.",
            },
        ],
    }
]

# Uncomment this row to use the above data instead of your evaluation set
eval_df = pd.DataFrame(eval_data)

# COMMAND ----------

# MAGIC %md
# MAGIC # Evaluate the POC application

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get the MLflow run of the POC application 

# COMMAND ----------

runs = mlflow.search_runs(experiment_names=[MLFLOW_EXPERIMENT_NAME], filter_string=f"run_name = '{POC_CHAIN_RUN_NAME}'", output_format="list")

if len(runs) != 1:
    print(f"Found {len(runs)} run with name {POC_CHAIN_RUN_NAME}.  Ensure the run name is accurate and try again.")

poc_run = runs[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the correct Python environment for the POC's app
# MAGIC
# MAGIC TODO: replace this with env_manager=virtualenv once that works

# COMMAND ----------

pip_requirements = mlflow.pyfunc.get_model_dependencies(f"runs:/{poc_run.info.run_id}/chain")

# COMMAND ----------

# MAGIC %pip install -r $pip_requirements

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run evaluation on the POC app

# COMMAND ----------

with mlflow.start_run(run_id=poc_run.info.run_id):
    # Evaluate
    eval_results = mlflow.evaluate(
        data=eval_df,
        model=f"runs:/{poc_run.info.run_id}/chain",  # replace `chain` with artifact_path that you used when calling log_model.  By default, this is `chain`.
        model_type="databricks-agent",
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Look at the evaluation results
# MAGIC
# MAGIC You can explore the evaluation results using the above links to the MLflow UI.  If you prefer to use the data directly, see the cells below.

# COMMAND ----------

# Summary metrics across the entire evaluation set
eval_results.metrics

# COMMAND ----------

# Evaluation results including LLM judge scores/rationales for each row in your evaluation set
per_question_results_df = eval_results.tables['eval_results']

# You can click on a row in the `trace` column to view the detailed MLflow trace
display(per_question_results_df)

# COMMAND ----------


