# Databricks notebook source
# MAGIC %pip install paperscraper
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import logging
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j").setLevel(logging.ERROR)

# COMMAND ----------

catalog = 'mlops_pj'
spark.sql(f'USE catalog {catalog} ')

schema  = 'rag_puneetjain'
spark.sql(f'USE schema {schema} ')

volume_folder = 'gsk_hackathon_test'
spark.sql(f'CREATE VOLUME IF NOT EXISTS {volume_folder} ')

# COMMAND ----------

from paperscraper.pubmed import get_and_dump_pubmed_papers
topics = [
         'CAR-T cell therapy',
        #  'cancer vaccines',
        #  'checkpoint inhibitors',
        #  'RNA therapeutics',
        #  'siRNA',
        #  'mRNA',
        #  'RNA interference',
        #  'gene therapy',
        #  'gene transfer',
        #  'tumor suppressor gene therapy',
        #  'epigenetic therapy',
        #  'histone modification',
        #  'chromatin remodelling',
        #  'HDAC DNMT inhibitors',
        #  'stem cell therapy',
        #  'MSCs',
        #  'cellular reprogramming',
        #  'ESCs',
        #  'iPSCs',
         'MSCs']




# COMMAND ----------

# import pandas as pd   
# get_and_dump_pubmed_papers(topic, output_filepath=f'/Volumes/{catalog}/{schema}/{volume_folder}/{topic[0]}.jsonl',max_results =200) 
# jsonObj = pd.read_json(path_or_buf=f'/Volumes/{catalog}/{schema}/{volume_folder}/{topic[0]}.jsonl', lines=True)


# COMMAND ----------

from util import *
import pandas as pd 

def search_and_download_paper_to_delta( topics,
                                        catalog,
                                        schema,
                                        volume_folder,
                                        max_results = 200,
                                        output_table = 'bronze_pubmed_pdf_list'):
  
  assert type(topics) == list, 'topic must be a list'
  
  for topic in topics:
    get_and_dump_pubmed_papers([topic], output_filepath=f'/Volumes/{catalog}/{schema}/{volume_folder}/{topic}.jsonl',max_results =max_results)

    jsonObj = pd.read_json(path_or_buf=f'/Volumes/{catalog}/{schema}/{volume_folder}/{topic}.jsonl', 
                           lines=True)
    
    jsonObj['pdf_output'] = jsonObj.apply(save_paper_pdf, axis=1, 
                                      catalog=catalog, 
                                      schema=schema, 
                                      volume_folder=volume_folder, 
                                      topic=topic)

    jsonObj['topic'] = topic
    df = spark.createDataFrame(jsonObj)
    df.write.format("delta").mode("append").saveAsTable(output_table)

  return df




# COMMAND ----------

search_and_download_paper_to_delta(
                                    topics,
                                    catalog,
                                    schema,
                                    volume_folder,
                                    output_table= 'bronze_pubmed_pdf_list')

# COMMAND ----------

# spark.sql("DROP TABLE IF EXISTS bronze_file_jump")

# COMMAND ----------

# from paperscraper.pdf import save_pdf
# paper_data = {'doi': jsonObj.iloc[0]['doi']}
# !mkdir -p /Volumes/{catalog}/{schema}/{volume_folder}/{topic[0]}/
# pdf = save_pdf(paper_data, filepath=f'/Volumes/{catalog}/{schema}/{volume_folder}/test.pdf')
# import os



# # Apply the function to each row in the DataFrame
# jsonObj['pdf_output'] = jsonObj.apply(save_paper_pdf, axis=1, 
#                                       catalog=catalog, 
#                                       schema=schema, 
#                                       volume_folder=volume_folder, 
#                                       topic=topic[0])

# jsonObj['topics'] = topic[0]


# COMMAND ----------




# COMMAND ----------

df = spark.sql("select * from test_pubmed_pdf_list")
# df = df.dropDuplicates()
# df.write.format("delta").mode("overwrite").saveAsTable("bronze_pubmed_pdf_list")

# COMMAND ----------

display(df)

# COMMAND ----------

filtered_df = df.filter(df.pdf_output.contains("/Volumes"))

# COMMAND ----------

display(filtered_df.count())

# COMMAND ----------

df.write.format("delta").mode("overwrite").saveAsTable("bronze_pubmed_pdf_list")

# COMMAND ----------


