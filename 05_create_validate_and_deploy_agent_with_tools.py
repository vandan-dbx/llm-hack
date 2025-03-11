# Databricks notebook source
# MAGIC %md
# MAGIC #Tool-calling Agent
# MAGIC
# MAGIC This is an auto-generated notebook created by an AI Playground export.
# MAGIC
# MAGIC This notebook uses [Mosaic AI Agent Framework](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/build-genai-apps) to recreate your agent from the AI Playground. It  demonstrates how to develop, manually test, evaluate, log, and deploy a tool-calling agent in LangGraph.
# MAGIC
# MAGIC The agent code implements [MLflow's ChatAgent](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ChatAgent) interface, a Databricks-recommended open-source standard that simplifies authoring multi-turn conversational agents, and is fully compatible with Mosaic AI agent framework functionality.
# MAGIC
# MAGIC  **_NOTE:_**  This notebook uses LangChain, but AI Agent Framework is compatible with any agent authoring framework, including LlamaIndex or pure Python agents written with the OpenAI SDK.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC
# MAGIC - Address all `TODO`s in this notebook.

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow langchain langgraph==0.2.76 databricks-langchain pydantic databricks-agents uv 
# MAGIC %pip install -U -qqqq git+https://github.com/unitycatalog/unitycatalog.git#subdirectory=ai/core&egg=unitycatalog-ai[databricks]
# MAGIC %pip install -U -qqqq git+https://github.com/unitycatalog/unitycatalog.git#subdirectory=ai/integrations/langchain&egg=unitycatalog-langchain[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md ## Define the agent in code
# MAGIC Below we define our agent code in a single cell, enabling us to easily write it to a local Python file for subsequent logging and deployment using the `%%writefile` magic command.
# MAGIC
# MAGIC For more examples of tools to add to your agent, see [docs](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/agent-tool).

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC from typing import Any, Generator, Optional, Sequence, Union
# MAGIC
# MAGIC import mlflow
# MAGIC from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool
# MAGIC from databricks_langchain.uc_ai import (
# MAGIC     DatabricksFunctionClient,
# MAGIC     UCFunctionToolkit,
# MAGIC     set_uc_function_client,
# MAGIC )
# MAGIC from langchain_core.language_models import LanguageModelLike
# MAGIC from langchain_core.runnables import RunnableConfig, RunnableLambda
# MAGIC from langchain_core.tools import BaseTool
# MAGIC from langgraph.graph import END, StateGraph
# MAGIC from langgraph.graph.graph import CompiledGraph
# MAGIC from langgraph.graph.state import CompiledStateGraph
# MAGIC from langgraph.prebuilt.tool_executor import ToolExecutor
# MAGIC from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
# MAGIC from mlflow.pyfunc import ChatAgent
# MAGIC from mlflow.types.agent import (
# MAGIC     ChatAgentChunk,
# MAGIC     ChatAgentMessage,
# MAGIC     ChatAgentResponse,
# MAGIC     ChatContext,
# MAGIC )
# MAGIC
# MAGIC mlflow.langchain.autolog()
# MAGIC
# MAGIC client = DatabricksFunctionClient()
# MAGIC set_uc_function_client(client)
# MAGIC
# MAGIC ############################################
# MAGIC #Define the UC Credentials 
# MAGIC ############################################
# MAGIC CATALOG = 'mlops_pj'
# MAGIC SCHEMA = 'hackathon'
# MAGIC
# MAGIC
# MAGIC ############################################
# MAGIC # Define your LLM endpoint and system prompt
# MAGIC ############################################
# MAGIC LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
# MAGIC llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC
# MAGIC system_prompt = ""
# MAGIC
# MAGIC ###############################################################################
# MAGIC ## Define tools for your agent, enabling it to retrieve data or take actions
# MAGIC ## beyond text generation
# MAGIC ## To create and see usage examples of more tools, see
# MAGIC ## https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/agent-tool
# MAGIC ###############################################################################
# MAGIC tools = []
# MAGIC
# MAGIC ##### You can use UDFs in Unity Catalog as agent tools ######
# MAGIC uc_tool_names = [f"{CATALOG}.{SCHEMA}.search_medical_leaflet",
# MAGIC                  f"{CATALOG}.{SCHEMA}.lookup_medicine_info",
# MAGIC                  f"{CATALOG}.{SCHEMA}.hackathon.lookup_drug_info"]
# MAGIC uc_toolkit = UCFunctionToolkit(function_names=uc_tool_names)
# MAGIC tools.extend(uc_toolkit.tools)
# MAGIC
# MAGIC #####################
# MAGIC ## Define agent logic
# MAGIC #####################
# MAGIC
# MAGIC
# MAGIC def create_tool_calling_agent(
# MAGIC     model: LanguageModelLike,
# MAGIC     tools: Union[ToolExecutor, Sequence[BaseTool]],
# MAGIC     system_prompt: Optional[str] = None,
# MAGIC ) -> CompiledGraph:
# MAGIC     model = model.bind_tools(tools)
# MAGIC
# MAGIC     # Define the function that determines which node to go to
# MAGIC     def should_continue(state: ChatAgentState):
# MAGIC         messages = state["messages"]
# MAGIC         last_message = messages[-1]
# MAGIC         # If there are function calls, continue. else, end
# MAGIC         if last_message.get("tool_calls"):
# MAGIC             return "continue"
# MAGIC         else:
# MAGIC             return "end"
# MAGIC
# MAGIC     if system_prompt:
# MAGIC         preprocessor = RunnableLambda(
# MAGIC             lambda state: [{"role": "system", "content": system_prompt}]
# MAGIC             + state["messages"]
# MAGIC         )
# MAGIC     else:
# MAGIC         preprocessor = RunnableLambda(lambda state: state["messages"])
# MAGIC     model_runnable = preprocessor | model
# MAGIC
# MAGIC     def call_model(
# MAGIC         state: ChatAgentState,
# MAGIC         config: RunnableConfig,
# MAGIC     ):
# MAGIC         response = model_runnable.invoke(state, config)
# MAGIC
# MAGIC         return {"messages": [response]}
# MAGIC
# MAGIC     workflow = StateGraph(ChatAgentState)
# MAGIC
# MAGIC     workflow.add_node("agent", RunnableLambda(call_model))
# MAGIC     workflow.add_node("tools", ChatAgentToolNode(tools))
# MAGIC
# MAGIC     workflow.set_entry_point("agent")
# MAGIC     workflow.add_conditional_edges(
# MAGIC         "agent",
# MAGIC         should_continue,
# MAGIC         {
# MAGIC             "continue": "tools",
# MAGIC             "end": END,
# MAGIC         },
# MAGIC     )
# MAGIC     workflow.add_edge("tools", "agent")
# MAGIC
# MAGIC     return workflow.compile()
# MAGIC
# MAGIC
# MAGIC class LangGraphChatAgent(ChatAgent):
# MAGIC     def __init__(self, agent: CompiledStateGraph):
# MAGIC         self.agent = agent
# MAGIC
# MAGIC     def predict(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> ChatAgentResponse:
# MAGIC         request = {"messages": self._convert_messages_to_dict(messages)}
# MAGIC
# MAGIC         messages = []
# MAGIC         for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC             for node_data in event.values():
# MAGIC                 messages.extend(
# MAGIC                     ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
# MAGIC                 )
# MAGIC         return ChatAgentResponse(messages=messages)
# MAGIC
# MAGIC     def predict_stream(
# MAGIC         self,
# MAGIC         messages: list[ChatAgentMessage],
# MAGIC         context: Optional[ChatContext] = None,
# MAGIC         custom_inputs: Optional[dict[str, Any]] = None,
# MAGIC     ) -> Generator[ChatAgentChunk, None, None]:
# MAGIC         request = {"messages": self._convert_messages_to_dict(messages)}
# MAGIC         for event in self.agent.stream(request, stream_mode="updates"):
# MAGIC             for node_data in event.values():
# MAGIC                 yield from (
# MAGIC                     ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
# MAGIC                 )
# MAGIC
# MAGIC
# MAGIC # Create the agent object, and specify it as the agent object to use when
# MAGIC # loading the agent back for inference via mlflow.models.set_model()
# MAGIC agent = create_tool_calling_agent(llm, tools, system_prompt)
# MAGIC AGENT = LangGraphChatAgent(agent)
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the agent
# MAGIC
# MAGIC Interact with the agent to test its output. Since this notebook called `mlflow.langchain.autolog()` you can view the trace for each step the agent takes.
# MAGIC
# MAGIC Replace this placeholder input with an appropriate domain-specific example for your agent.

# COMMAND ----------

from agent import AGENT

AGENT.predict({"messages": [{"role": "user", "content": "Who should avoid taking Amoxicillin capsules?"}]})

# COMMAND ----------

for event in AGENT.predict_stream(
    {"messages": [{"role": "user", "content": "What is 5+5 in python"}]}
):
    print(event, "-----------\n")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Log the `agent` as an MLflow model
# MAGIC Determine Databricks resources to specify for automatic auth passthrough at deployment time
# MAGIC - **TODO**: If your Unity Catalog tool queries a [vector search index](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/unstructured-retrieval-tools) or leverages [external functions](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/external-connection-tools), you need to include the dependent vector search index and UC connection objects, respectively, as resources. See [docs](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/log-agent#specify-resources-for-automatic-authentication-passthrough) for more details.
# MAGIC
# MAGIC Log the agent as code from the `agent.py` file. See [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).

# COMMAND ----------

# Determine Databricks resources to specify for automatic auth passthrough at deployment time
import mlflow
from agent import tools, LLM_ENDPOINT_NAME
from databricks_langchain import VectorSearchRetrieverTool
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint,DatabricksTable
from unitycatalog.ai.langchain.toolkit import UnityCatalogTool

# TODO: Manually include underlying resources if needed. See the TODO in the markdown above for more information.
resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME)]
for tool in tools:
    if isinstance(tool, VectorSearchRetrieverTool):
        resources.extend(tool.resources)
    elif isinstance(tool, UnityCatalogTool):
        resources.append(DatabricksFunction(function_name=tool.uc_function_name))

# **** Ensure all the relevant tables have been provided ****
table_list = ['nmanchev.pharma.medicine_leaflets_chunked_index',
              'nmanchev.pharma.drugs_com',
              'nmanchev.pharma.medicine'] 

for table in table_list:
    resources.append(DatabricksTable(table_name=table))

input_example = {
    "messages": [
        {
            "role": "user",
            "content": "What is an LLM agent?"
        }
    ]
}

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model="agent.py",
        input_example=input_example,
        pip_requirements=[
            "mlflow",
            "langchain",
            "langgraph==0.2.76",
            "databricks-langchain", 
            "git+https://github.com/unitycatalog/unitycatalog.git#subdirectory=ai/core&egg=unitycatalog-ai[databricks]",
            "git+https://github.com/unitycatalog/unitycatalog.git#subdirectory=ai/integrations/langchain&egg=unitycatalog-langchain[databricks]",
            "pydantic"
        ],
        resources=resources,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the agent with [Agent Evaluation](https://learn.microsoft.com/azure/databricks/generative-ai/agent-evaluation/)
# MAGIC
# MAGIC You can edit the requests or expected responses in your evaluation dataset and run evaluation as you iterate your agent, leveraging mlflow to track the computed quality metrics.
# MAGIC
# MAGIC To evaluate your tool calls, try adding [custom metrics](https://learn.microsoft.com/azure/databricks/generative-ai/agent-evaluation/custom-metrics#evaluating-tool-calls).

# COMMAND ----------

import pandas as pd

# eval_examples = [
#     {
#         "request": {
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": "What is an LLM agent?"
#                 }
#             ]
#         },
#         "expected_response": None
#     }
# ]

eval_dataset = spark.read.table("mlops_pj.hackathon.pharma_agent_eval_set").toPandas() 
display(eval_dataset)


# COMMAND ----------

import mlflow

with mlflow.start_run(run_id=logged_agent_info.run_id):
    eval_results = mlflow.evaluate(
        f"runs:/{logged_agent_info.run_id}/agent",
        data=eval_dataset,  # Your evaluation dataset
        model_type="databricks-agent",  # Enable Mosaic AI Agent Evaluation
    )

# Review the evaluation results in the MLFLow UI (see console output), or access them in place:
display(eval_results.tables['eval_results'])

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Perform pre-deployment validation of the agent
# MAGIC Before registering and deploying the agent, we perform pre-deployment checks via the [mlflow.models.predict()](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.predict) API. See [documentation](https://learn.microsoft.com/azure/databricks/machine-learning/model-serving/model-serving-debug#validate-inputs) for details

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agent",
    input_data={"messages": [{"role": "user", "content": "Hello!"}]},
    env_manager="uv"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC
# MAGIC Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.

# COMMAND ----------

from cookbook.config.shared.agent_storage_location import AgentStorageConfig
from cookbook.databricks_utils import get_table_url
from cookbook.config import load_serializable_config_from_yaml_file

# Load the Agent's storage configuration
agent_storage_config: AgentStorageConfig = load_serializable_config_from_yaml_file('./configs/agent_storage_config.yaml')

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=agent_storage_config.uc_model_name
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent

# COMMAND ----------

from databricks import agents
agents.deploy(gent_storage_config.uc_model_name
), uc_registered_model_info.version, tags = {"endpointSource": "playground"})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps
# MAGIC
# MAGIC After your agent is deployed, you can chat with it in AI playground to perform additional checks, share it with SMEs in your organization for feedback, or embed it in a production application. See [docs](https://learn.microsoft.com/azure/databricks/generative-ai/deploy-agent) for details

# COMMAND ----------


