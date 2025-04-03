# Databricks notebook source
# MAGIC %md 
# MAGIC ### These are boiler plate code to create UC Functions which can be integrated as tools

# COMMAND ----------

# MAGIC  %pip install -qqqq -U -r requirements.txt
# MAGIC  dbutils.library.restartPython()

# COMMAND ----------

from cookbook.config.shared.agent_storage_location import AgentStorageConfig
from cookbook.config.data_pipeline import (
    DataPipelineConfig,
)
from cookbook.databricks_utils import get_table_url, get_current_user_info
from cookbook.config import load_serializable_config_from_yaml_file

# Load the Agent's storage configuration
data_pipeline_config: DataPipelineConfig = load_serializable_config_from_yaml_file(
    "./configs/data_pipeline_config.yaml"
)


# COMMAND ----------

user_email, user_name, default_catalog = get_current_user_info(spark)

# COMMAND ----------

CATALOG = "users"
SCHEMA = user_name
INDEX_NAME = f"{CATALOG}.{SCHEMA}.medicine_leaflets_chunked_index"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create function to gather reviews about the drug

# COMMAND ----------

spark.sql(
    f"""
          CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.lookup_drug_info(drug_name STRING COMMENT
          'Name of the drug whose information to look up')
          RETURNS TABLE(drugName STRING , condition STRING, review STRING,rating INT, usefulCount int ,date STRING)
          COMMENT "Returns the top 10 reviews for a drug (from the drugs_com table), ordered by usefulness"
          RETURN SELECT drugName,condition,review,rating,usefulCount,date
          FROM workspace.shared.drugs 
          WHERE drugName like concat("%",drug_name,"%")
          ORDER BY usefulCount DESC
          LIMIT 1
          """
)

# COMMAND ----------

spark.sql(
    f"""
            CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.lookup_drug_info(
            drug_name STRING COMMENT 'Name of the drug whose information to look up')
            RETURNS TABLE(drugName STRING , condition STRING, review STRING,rating INT, usefulCount int ,date STRING)
            COMMENT "Returns the top 10 reviews for a drug (from the drugs_com table), ordered by usefulness"
            RETURN SELECT drugName,condition,review,rating,usefulCount,date
            FROM workspace.shared.drugs 
            WHERE drugName like concat("%",drug_name,"%")
            ORDER BY usefulCount DESC
            LIMIT 1
          """
)

# COMMAND ----------

spark.sql(f"""
          CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.lookup_drug_info(
  drug_name STRING COMMENT 'Name of the drug whose information to look up')
RETURNS TABLE(drugName STRING , condition STRING, review STRING,rating INT, usefulCount int ,date STRING)
COMMENT "Returns the top 10 reviews for a drug (from the drugs_com table), ordered by usefulness"
RETURN SELECT drugName,condition,review,rating,usefulCount,date
  FROM workspace.shared.drugs 
  WHERE drugName like concat("%",drug_name,"%")
  ORDER BY usefulCount DESC
  LIMIT 1
  """)

# COMMAND ----------

spark.sql(
    f"""
        CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.lookup_medicine_info(
    medicine_name STRING COMMENT 'Name of the medicine whose information to look up')
    RETURNS STRING
    COMMENT "Returns concatenated medicine information"
    RETURN select concat("name: ",name,
             " ,substitute0: ",COALESCE(substitute0,""),
             " ,use0: ",COALESCE(use0,"")) from workspace.shared.medicine where name like concat("%",lower(medicine_name),"%") LIMIT 1"""
)

# COMMAND ----------

# MAGIC %sql
# MAGIC select  concat("name: ",name,
# MAGIC              " ,substitute0: ",COALESCE(substitute0,""))
# MAGIC   from workspace.shared.medicine LIMIT 1

# COMMAND ----------

spark.sql(
    f"""
    CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.search_medical_leaflet (
  -- The agent uses this comment to determine how to generate the query string parameter.
  query STRING
  COMMENT 'The query string for searching medical leaflets.'
) RETURNS TABLE
-- The agent uses this comment to determine when to call this tool. It describes the types of documents and information contained within the index.
COMMENT 'Executes a search on medical leaflets to retrieve the most relevant text documents based on the input query.'
RETURN SELECT
  content_chunked AS page_content,
  map('doc_uri', doc_uri, 'chunk_id', chunk_id) AS metadata
FROM
  vector_search(
    -- Specify your Vector Search index name here
    index => "{INDEX_NAME}",
    query => query,
    num_results => 1
  )
"""
)
