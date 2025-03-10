import pandas as pd 
import numpy as np 
import seaborn as sns 
import ast , json
from matplotlib.pyplot import plot as plt 
from langchain_chroma.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, WebBaseLoader , TextLoader
from langchain_text_splitters.sentence_transformers import SentenceTransformersTokenTextSplitter
from langchain.agents import LLMSingleActionAgent, create_openai_tools_agent, create_react_agent, create_openai_functions_agent , ConversationalAgent 
from langchain.memory import ConversationBufferWindowMemory 
from langchain_core.documents import Document
from langchain.chains import AnalyzeDocumentChain
from pdf2image import convert_from_path
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter , TextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor , create_react_agent, create_openai_functions_agent, create_openai_tools_agent
from langchain.chains import LLMChain
from langchain.tools import Tool
from pymongo.mongo_client import MongoClient
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from pydantic import BaseModel, conlist
from bson import ObjectId
import re
from langchain.schema.runnable import RunnableSequence

memory = ConversationBufferWindowMemory(memory_key="chat_history",return_messages=True,k=3)



## Pre Processing of the Data like Documents, Web Scraping, Loading the Data, Vector Databae and Retrivers....
paths = ["Tesla Model 3.pdf"] #"Tesla Model S Manual.pdf","Tesla Model X Manual.pdf","Tesla Model y Manual.pdf"]

    
def get_extract_content(pdf_path):
    text = ""
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
            tables.extend(page.extract_tables())
    return text, tables


# Function to load PDFs and process them into documents for vectorization
def load_pdf(pdf_paths):
    all_documents = []
    for pdf_path in pdf_paths:
        text, tables = get_extract_content(pdf_path)
        full_text = f"{text}\nTables: {str(tables)} "  # images can be added later if needed
        vector_store = text_splitter(full_text)
        print("Hre is the vector store:- ",vector_store)
        all_documents.append(vector_store)
    print("Here is the All Documents:- ",all_documents)
    return all_documents


# Function to split the text into smaller chunks for embeddings
def text_splitter(full_text):
    text_splitter_instance = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    chunks = text_splitter_instance.split_text(full_text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    embedder = OpenAIEmbeddings(model="text-embedding-3-small", api_key=open_api_key)  # Use your actual API key
    vector_store = FAISS.from_documents(documents, embedder)
    return vector_store


# Function to create a retriever from the vector store


# Example usage:
paths = ["Tesla Model 3.pdf"]  # Replace with actual paths to your PDFs
documents = load_pdf(paths)

# Query to retrieve relevant information
vector_st = documents[0]
retriever = vector_st.as_retriever(search_type="similarity", search_kwargs={"k": 5})

def get_retriever(query)-> str:
    resposne = retriever.get_relevant_documents(query)
    print("Hree is the final retrivever:- ",resposne)
    return resposne









### Master Agent Prompt ...

master_prompt = """You are a Tesla Virtual Assistant, integrated into Tesla’s AI-powered Enterprise Resource Planning (ERP) system. Your role is to assist Tesla employees by automating workflows, providing real-time insights, and streamlining operations across key business functions, including Inventory Management, Supply Chain, Manufacturing, Engineering, Finance, and Sales.
You have deep expertise in Tesla’s internal processes, data structures, and enterprise operations. Your goal is to reduce manual work, increase efficiency, and enable data-driven decision-making for employees.
Restrictions & Instructions
 DO NOT:
 Provide any programming code unless explicitly requested.
 Generate long, unnecessary explanations—be concise and to the point.
 Offer external links unless related to internal Tesla ERP documentation.
 Give unauthorized financial disclosures or sensitive business insights.
 MUST DO:
 Keep responses brief, clear, and structured unless the query requires additional depth.
 If the employee’s request is too broad, suggest a more specific approach or ask clarifying questions.
 Provide more context only if it improves understanding or aids decision-making.
 Format responses professionally, using bullet points, tables, or key highlights where applicable.
🎯 Your Responsibilities & Examples
1️⃣ Inventory Management
🔹 Monitor stock levels in real-time across Tesla Gigafactories and Service Centers.
 🔹 Predict and alert about critical component shortages (e.g., battery cells, semiconductor chips).
 🔹 Automate purchase order creation for restocking essential parts.
Example Query:
 🗨️ "Check the stock levels of Model 3 battery modules in Gigafactory Nevada and suggest a reorder date."
2️⃣ Supply Chain Management
🔹 Track shipment status of essential materials from Tesla’s global suppliers.
 🔹 Identify delays or risks and suggest alternative supply chain strategies.
 🔹 Monitor supplier performance and contract compliance.
Example Query:
 🗨️ "Provide an update on the lithium-ion battery shipment from Shanghai and estimated delivery time."
3️⃣ Manufacturing & Production
🔹 Analyze real-time factory production efficiency and identify bottlenecks.
 🔹 Predict maintenance needs to prevent unexpected machine downtime.
 🔹 Provide quality control reports to ensure Tesla’s production standards.
Example Query:
 🗨️ "What is the defect rate of the latest Model S production batch, and how can we improve it?"
4️⃣ Engineering Support
🔹 Retrieve technical documentation, design blueprints, and testing reports.
 🔹 Automate design validation checks for vehicle models and software updates.
 🔹 Provide AI-driven insights into energy efficiency and vehicle performance.
Example Query:
 🗨️ "Fetch the aerodynamic simulation report for the new Roadster and highlight efficiency improvements."
5️⃣ Finance & Accounting
🔹 Automate payroll processing, expense approvals, and budget tracking.
 🔹 Generate financial forecasts, revenue reports, and tax compliance summaries.
 🔹 Identify cost-saving opportunities using AI-powered analytics.
Example Query:
 🗨️ "Generate a quarterly financial report for Q1 2025 with detailed expense breakdowns."
6️⃣ Sales & Business Intelligence
🔹 Analyze Tesla’s global sales data and revenue trends.
 🔹 Predict demand fluctuations and suggest production adjustments.
 🔹 Provide lead conversion analysis for Tesla’s sales and marketing teams.
Example Query:
 🗨️ "What are the projected Model Y sales in the European market for the next quarter?"
🎯 Response Format & Guidelines
✅ Use professional and structured formatting:
📊 Data Tables for numerical insights.
🔹 Bullet Points for clarity.
📈 Charts/Graphs (if applicable) for trend analysis.
✅ Clarify when needed:
If a query is vague, respond with:
 🗨️ "Can you specify the location or department for a more accurate response?"
✅ Be proactive & suggest automation:
Instead of just answering, offer automation ideas to improve efficiency.
⚡ Example Interaction:
🧑 Tesla Employee:
 "Check the current stock of Tesla Supercharger components and estimate when we need to reorder."
🤖 Tesla AI Assistant:
 📊 Current Supercharger Component Stock (Gigafactory Nevada):
Component
In Stock
Estimated Reorder Date
Power Modules
1200
March 15, 2025
Cooling Units
800
February 28, 2025
Charging Cables
2500
April 5, 2025

🔹 Based on current usage trends, I recommend placing a reorder for Cooling Units by February 20, 2025.
🔹 Would you like to automate this reorder process? ✅❌

Tools You Have:

retrieval: Use this tool only when extracting information about Tesla cars from a customer perspective, such as specifications, speed, features, and part details. This is useful for answering general queries about Tesla models and their performance.
db_mongo: Use this tool whenever retrieving structured data from the database, including sales, service records, inventory, production details, supply chain, transactions, and historical data. This tool is ideal for handling business-related queries that require real-time or historical data analysis.
Pass the Same query to all tools you get from the user. 
You are an advanced AI data analyst with expertise in ERP systems, specifically Tesla’s ERP. After Geting the Data from db_mongo, Analyse the Data, Provide 
    Overall Business Performance

    Key sales trends
    Most profitable Tesla products
    Revenue and cost analysis
    Supply Chain & Inventory Insights

    Identify bottlenecks in production
    Inventory shortages and their impact
    Supplier performance
    Production Efficiency

    Factory utilization & delays
    Comparison of different Tesla manufacturing plants
    Customer & Market Analysis

    Demand trends by region
    Customer complaints & warranty trends
    Actionable Recommendations

    Strategies for reducing costs and increasing efficiency
    Inventory management improvements
    Enhancing customer satisfaction

for example :-
    Input : Provide me the car of this customer id CUST-2FE819C3
    call the db_mongo tool
    Input : What's specifications of Tesla Model S and Model Y
    call the retrieval tool 
    Input : Provide me last year Model S Sales data
    call the db_mongo

    Input: Generate the Sale Report 
    Output: Sales Report:
            Total Units Sold: 23,343
            Revenue Generated: $2,334,343,400
            Highest Sales Region: Houston
            Lowest Sales Region: Texas
            Cost of Goods Sold: $1,923,234,300
            Profit Margin: $411,109,100
            Current Stock in Inventory: 2,323 units
            Reserved Cars: 2,233 units
            
    Input:  Let me know about the Sales Trend and which Model we need in stock for next 3 months.
    Output: 🚗 Sales Trend Analysis (Last 12 Months)
            Based on historical sales data from January 2023 to February 2024, we have observed the following key trends:
            Model Y & Model 3 Dominate Sales

            Model Y continues to be the best-selling vehicle, with demand rising 15% QoQ (Quarter-over-Quarter).
            Model 3 has stable sales, showing consistent customer preference for affordability & range.
            Seasonal Demand Patterns

            Higher demand in Q4 due to holiday & year-end discounts.
            Slight dip in Q1, but March tends to see a pre-summer sales boost.
            EV Incentives & Market Shift

            Increased government EV subsidies in some regions have driven sales upward.
            Model S & Model X saw lower demand due to market shifts towards affordable models.


            Stock Forecast for Next 3 Months (March - May 2024)
            🔹 Priority Stock Requirements:

            Model Y → Increase Stock by 20% 📈 (High Demand & Market Preference)
            Model 3 → Maintain Current Stock (Stable Demand)
            Model S → Reduce Stock by 10% (Lower Demand in the Luxury Segment)
            Model X → Maintain Minimum Stock (Selective Buyers)         

            🛠️ Suggested Actions
            ✅ Increase production & stock allocation for Model Y to meet rising demand.
            ✅ Monitor real-time orders for Model 3 to prevent stock shortages.
            ✅ Adjust Model S inventory based on premium buyer demand trends.
            ✅ Offer limited-time incentives on Model X to boost sales.

        Input: What’s the risk of a shortage for lithium-ion batteries next quarter?
        Output:
            Based on current market trends, there is a 20% risk of a lithium-ion battery shortage next quarter. This is primarily driven by rising global demand, supply chain constraints, and increasing raw material costs. To mitigate potential disruptions and ensure steady supply, it is advisable to place a bulk order in advance. Doing so can help secure inventory, minimize procurement risks, and potentially reduce costs by 10% through strategic purchasing and supplier negotiations.

        Input: Which suppliers can deliver the fastest without increasing costs?    ['Retrive Car_ID from Sales_Order_Data then get the Salesperson name from Customer_Data']
        Output:
        Based on supplier performance data, Tracy Hahn can fulfill the order within 3 days at a rate of $50 per unit, ensuring the fastest delivery. Jacob Bell offers a slightly lower cost of $45 per unit but requires 5 days for delivery.
        To meet urgent deadlines and avoid potential delays, it is recommended to proceed with Tracy Hahn for faster fulfillment. However, if cost savings take priority over speed, Jacob Bell remains a viable alternative

Note: Try to keep Response under the 200000 Tokens
        
User query:
{input}
"""

class MongoDBQuery(BaseModel):
    operation: str
    collection: str
    filter: dict = {}
    projection: dict = {}
    pipeline: list = []
    lookup: dict = {}


class MongoDBExecutor:
    def __init__(self, uri, db_name="Tesla_Data"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

    def execute_query(self, query):
        print("In execute query")
        try:
            print("Pinged your deployment. You successfully connected to MongoDB!")
            collection = self.db[query['collection']]
            
            if query['operation'] == "find":
                print("In if")
                result = collection.find(
                    query.get("filter", {}),
                    query.get("projection", {})
                ).limit(20).max_time_ms(60000)
                return list(result)

            elif query['operation'] == 'aggregate':
                pipeline = query.get("pipeline", [])
                print("In elif")
                # Add $lookup ONLY if all required fields are present
                if query.get("lookup") and all(
                    key in query['lookup'] for key in ["from", "localField", "foreignField", "as"]
                ):
                    print("In elif if")
                    pipeline.append({
                        "$lookup": {
                            "from": query['lookup']['from'],
                            "localField": query['lookup']['localField'],
                            "foreignField": query['lookup']['foreignField'],
                            "as": query['lookup']['as']
                        }
                    })
                print("Her eis the pipeline ",pipeline)
                # print("The query passed to mongo db :-",list(collection.aggregate(pipeline,maxTimeMS=500000)))
                return list(collection.aggregate(pipeline, maxTimeMS=60000))
            
            else:
                raise ValueError("Unsupported operation")
            
        except Exception as e:
            return {"error": str(e)}




### Defne the tools functions here 

def mongo_engine(query) ->str:


    # print("In mongo query",query['text'])
    # pattern = r"^```(?:javascript)?\s*\n(.*?)\n```$"
    # match = re.search(pattern, query['text'], re.DOTALL)
    # if match:
    #     query = match.group(1)
    # else:
    #     query
    print("In mongo")
    print("In the Mongo Engine,",query)
    print("Type of query:- ",type(query))
    # Db connections 
    uri = "mongodb+srv://harsimran726:12Code12%2B@cluster0.pp6hs.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    if 'json' in query:
        query = query[8:-3]
        query = ast.literal_eval(query)
        print(type(query))
        print("Query after json remvoed",query)
    else:
        print("In else")
        # query = ast.literal_eval(query)
        query = json.loads(query)
        print("After laods :- ",query)
    try: 
        print("In try engine, ",type(query))
        validated_query = MongoDBQuery(**query).dict()
        print("In try 1")
    except Exception as e:
        return f"Invalid query {str(e)}"
    
    executer = MongoDBExecutor(uri)
    print("In try 2")
    result = executer.execute_query(validated_query)
    print("Here is the mongo result:- ",result)

    return result


def mongodb_expert(query)->str:
    """MongoDB Expert with 10 years of expertise.. """

    db_prompt ="""You are an expert MongoDB query generator with more than 5 yeas experience. Your task is two-fold:
Understand the User Query:
Read and analyze the given plain-English query, identify the key requirements, and determine which collections and fields are involved.
Consider operations like filtering, grouping, sorting, mathematical aggregations (e.g., sum, average, count), and even join-like operations using the MongoDB aggregation pipeline (e.g., $lookup).

Generate a NoSQL Script:
Convert the understood query into a fully functional MongoDB NoSQL script. Use the appropriate commands (find, aggregate, update, etc.) and aggregation operators (such as $match, $group, $lookup, $project, etc.) to extract or manipulate data accordingly.

Database Structure & Collections:

Car_Data

Fields:
Car_ID (e.g., "2024ModelSModelSB&001")
Model (e.g., "Model S")
Variant (e.g., "Model S")
Colour (e.g., "Ultra Red")
Manufacturing Year (e.g., 2024)
Status (e.g., "Sold")
Features (e.g., "Black & White Interior, York Steering")
Price (e.g., 76490)
Customer_Data

Fields:
Customer_ID (e.g., "CUST-2FE819C3")
Full Name (e.g., "Stephanie Alexander")
Email (e.g., "gentryamber@example.com")
Phone Number (e.g., "400-959-1028")
Address (e.g., "7658 Billy Hollow Suite 900, East Marieside, CT 01918")
Order Date (e.g., ISODate("2024-10-17T00:00:00.000+00:00"))
Car_ID
Salesperson (e.g., "Amy Johnson")
Order Status (only allowed values: Delivered, In Transit, Pending)
Payment Method (e.g., "Financing")
Payment Status (Paid, Pending)
Discount Applied (Boolean)
Warranty Period (years) (e.g., 1)
Service Contracts (e.g., "None")
Inventory_Data

Fields:
Model
Variant
Color (e.g., "Ultra Red")
Manufacturing Year
Price
Stock Quantity
Arrival Date (e.g., ISODate("2024-06-02T00:00:00.000+00:00"))
Location (e.g., "Unit 0003 Box 7003 DPO AA 72321")
Supplier (from supplier list)
Features
Battery Capacity (kWh) (e.g., 250)
Status (e.g., "Sold")

Raw_material

Fields:
Material Name
Usage
Car Model (e.g., Model S, Model 3, Model X, Model Y, Cybertruck, Roadster)
Variant (e.g., Long Range, Plaid, Standard Range Plus, Performance, etc.)
date_time_in: When the material enters the inventory
date_time_out: When the material is sent out for production
total_comes: Total quantity received into inventory
total_output: Total quantity output to production
how_much_require: The amount of that material required for one car

part_components

Fields:
Part Name: The identifier for the specific part or component.
Category: The classification of the part (e.g., Battery & Powertrain, Chassis & Body).
Usage: A brief description of how the part is used in production.
Price_USD: The unit price of the part in U.S. dollars.
COGS_USD: The cost of goods sold per unit in U.S. dollars.
date_time_in: The timestamp when the part enters the inventory.
date_time_out: The timestamp when the part is dispatched for production.
total_comes: The total quantity of the part received into inventory.
total_goes: The total quantity of the part that left inventory (used in production).

Sales_Order_Data

Fields:
Order_ID (e.g., "ORD-00001")
Customer_ID
Car_ID
Order Date (e.g., ISODate("2024-05-23T00:00:00.000+00:00"))
Delivery Date (e.g., ISODate("2024-05-30T00:00:00.000+00:00"))
Order Status (Delivered, In Transit, Pending)
Payment Status (Paid, Pending)
Total Amount
Discounts
Salesperson ID (e.g., "SP-050")
Service_Data

Fields:
Service_ID (e.g., "SVC-00001")
Car_ID
Service Date (e.g., ISODate("2024-05-13T00:00:00.000+00:00"))
Service Type (e.g., "Alignment")
Service Center (e.g., "Harrell, Riddle and Robinson - Port Nicole")
Service Status (e.g., "Pending")
Service Cost (e.g., 1750)
Technician Name (choose from: Danielle Duarte, Margaret Mitchell, Ashley Morgan, Russell Stewart, Jennifer Robertson, Jacob Olsen, Luis Cruz, Amber Boyd, Kim Jacobson, Chelsea Scott)
Parts Replaced (e.g., "Radiator, Timing Belt")
Warranty Claim (e.g., "No")
Supply_Chain

Fields:
Supplier_ID (UUID format)
Supply Chain Stage (e.g., "Transit")
Manufacturing Location (choose from: "Shanghai, China", "Fremont, California", "Sparks, Nevada", "Berlin, Germany", "Grünheide, Germany", "Austin, Texas", "Buffalo, New York")
Supplier Name (choose from: "Strickland-Anderson", "Pierce-Brown", "Meyer, Webb and Ponce", "Joseph, Arnold and Stewart", "Jones-Kaufman", "Owens-Richardson", "Jimenez, Hurst and Brown", "Williams-Mosley", "Martin, Garcia and Strickland", "Soto Inc")
Shipment Date (e.g., ISODate("2024-06-04T00:00:00.000+00:00"))
Shipment Mode (Sea, Air, Land)
Delivery Date (e.g., ISODate("2024-06-22T00:00:00.000+00:00"))
Shipping Cost (e.g., 1159.62)
Customs Clearance (e.g., "Pending")
Lead Time (e.g., 18 days)
Stock at Supplier (e.g., 49)
Transit Duration (e.g., 14 days)
Car_ID
Transactions_Data

Fields:
Transaction_ID (UUID format)
Transaction Date (e.g., ISODate("2024-05-25T00:00:00.000+00:00"))
Amount (e.g., 76490)
Transaction Type (e.g., "Payment")
Customer_ID
Car_ID
Payment Status (Paid, Pending)
Cost of Goods Sold (COGS) (e.g., 53543)
Profit Margin (e.g., 22947)
Additional Constraints & Data Points:

Supplier List:
Freeman, Lang and Contreras; Smith, Graves and Barrett; Mcdonald, Torres and Holland; Roberts-Lopez, James Inc; Johnston Ltd; Thomas, Burch and Holden; Morris, Holt and Parker; Fernandez, Robertson and Francis; Rasmussen, Pham and Herrera.

Parts & Components:
Lithium-ion Battery Pack
Electric Motor
Inverter
Chassis
Infotainment System

Material Names (Raw Materials):
Aluminum
Steel
Copper
Nickel
Cobalt
Lithium
Graphite
Silicon
Rare Earth Metals
Plastic & Composites
Glass
Rubber
Adhesives & Sealants

Allowed Values:

Order_Status: Delivered, In Transit, Pending
Payment_Status: Paid, Pending
Technician Name: [Danielle Duarte, Margaret Mitchell, Ashley Morgan, Russell Stewart, Jennifer Robertson, Jacob Olsen, Luis Cruz, Amber Boyd, Kim Jacobson, Chelsea Scott]
Manufacturing Location: ["Shanghai, China", "Fremont, California", "Sparks, Nevada", "Berlin, Germany", "Grünheide, Germany", "Austin, Texas", "Buffalo, New York"]
Supplier Name: ["Strickland-Anderson", "Pierce-Brown", "Meyer, Webb and Ponce", "Joseph, Arnold and Stewart", "Jones-Kaufman", "Owens-Richardson", "Jimenez, Hurst and Brown", "Williams-Mosley", "Martin, Garcia and Strickland", "Soto Inc"]
Shipment Mode: Sea, Air, Land
Instructions for Aggregations & Examples:

Aggregation Examples:

Example 1:
"Group by Manufacturing Year in Car_Data to calculate the total count of cars sold and the sum of their prices."
(This could use a pipeline with $match for Status: "Sold", then $group to count and sum.)

Example 2:
"Join Sales_Order_Data with Customer_Data based on Customer_ID and then group by Salesperson to compute the average sales amount."
(This would use $lookup to join the collections, followed by $group to perform the calculations.)

Your Task:

When provided with a query (e.g., “Find the total number of cars sold per model in 2024 along with the average selling price, and list the details of the corresponding salesperson from the customer data”), break it down into parts.
Identify the relevant collections (Car_Data, Sales_Order_Data, and Customer_Data).
Use appropriate MongoDB operations (such as $match, $group, $lookup, $project, and math operators like $sum, $avg, etc.) to create an aggregation pipeline that addresses the query.
Ensure the final output script is correct, clear, and ready to execute in a MongoDB environment.
Final Prompt Instruction:
Given the above database schema and constraints, convert the following user query into a complete MongoDB NoSQL script:
Note: Don't use ' ISODate("2023-01-01T00:00:00.000+00:00")' instead use '2023-01-01T00:00:00.000Z'.
      Don't provide comments in script
Output MUST be in this EXACT JSON format and only return the Script:
{{
    "operation": "find|aggregate",
    "collection": "Customer_Data|Car_Data|...",
    "filter": {{"key": "value"}},
    "projection": {{"field": 1}},
    "pipeline": [],
    "lookup": {{
        "from": "CollectionName",
        "localField": "field",
        "foreignField": "field",
        "as": "alias"
    }}
}}

Example Response:
{{
    "operation": "aggregate",
    "collection": "Customer_Data",
    "filter": {{"Customer_ID": "CUST-123"}},
    "projection": {{"Car_ID": 1, "_id": 0}},
    "pipeline": [],
    "lookup": {{
        "from": "Car_Data",
        "localField": "Car_ID",
        "foreignField": "Car_ID",
        "as": "VehicleDetails"
    }}
}}
Example Response:
{{
    "operation": "aggregate",
    "collection": "Car_Data",
    "filter": {{}},
    "projection": {{}},
    "pipeline": [
        {{
            "$match": {{
                "Model": {{ "$in": ["Model S", "Model Y"] }},
                "Status": {{'Sold}},
            }}
        }},
        {{
            "$project": {{
                "Car_ID": 1,
                "Model": 1,
                "Variant": 1,
                "Price": 1,
                "_id": 0
            }}
        }}
    ]
}}

"""


    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                db_prompt
            ),
             ("placeholder","{chat_history}"),
            HumanMessagePromptTemplate.from_template("{input}"),
            # MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    print("Here is the query:- ",query)
    llm = ChatOpenAI(model="gpt-4o-mini",api_key=open_api_key)
    chain = RunnableSequence(prompt | llm)

    response = chain.invoke({"input": query})

    print("HEre is the resposne:- ",response.content)


    

    mongo_response = mongo_engine(response.content)
    # final_response = mean_info_pro(mongo_response)
    # print("Response got after mogno db",final_response)
    # print("Final response got:- ",response)
    # for docu in response:
    #     docs.append(docu)
    #     print("Here is the mogno db fina l reponse :- ",docu)
    return mongo_response



def mean_info_pro(query) -> str:
    master_prompt = """
    You are an advanced AI data analyst with expertise in ERP systems, specifically Tesla’s ERP. Given the following dataset, analyze the data and extract meaningful insights about Tesla’s business operations, supply chain, inventory management, financial performance, and production efficiency. Provide a structured summary with key observations, trends, and potential recommendations for optimization.
    Output Requirements:
Overall Business Performance

Key sales trends
Most profitable Tesla products
Revenue and cost analysis
Supply Chain & Inventory Insights

Identify bottlenecks in production
Inventory shortages and their impact
Supplier performance
Production Efficiency

Factory utilization & delays
Comparison of different Tesla manufacturing plants
Customer & Market Analysis

Demand trends by region
Customer complaints & warranty trends
Actionable Recommendations

Strategies for reducing costs and increasing efficiency
Inventory management improvements
Enhancing customer satisfaction

Always output the Data (Given to you for Analyse ) in Table Format.
"""

    print("In mean info")
    try:    
        print("In meAN NFO 2")
        prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                master_prompt
            ),
             ("placeholder","{chat_history}"),
            ("human","{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
        )
        llm = ChatOpenAI(model="gpt-4o-mini",api_key=open_api_key,temperature=0.9)

        agent_chain = RunnableSequence(prompt | llm)
        response = agent_chain.invoke({"input":query})
        print("Succesfully completed")
        return response
    
    except Exception as e:
        response = f"An error occurred: {e}"
    



prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            master_prompt
        ),
         ("placeholder","{chat_history}"),
        ("user","{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ],
)







llm = ChatOpenAI(model="gpt-4o-mini",api_key=open_api_key,temperature=0.8)


## Tools define below ... 

retrieval = Tool(
    name="Retriever_Tool",
    func=get_retriever,
    description="Use this tool to retrieve information from the vector store related to Tesla cars, including specifications, brochures, performance details, features, and customer-oriented insights."
)

db_mongo = Tool(
    name="Mongo_DB",
    func=mongodb_expert,
    description="Use this tool to extract structured data from the database, including sales, service records, inventory, production details, supply chain, transactions, and other business-related information."
)


information_provider = Tool(
    name="information_provider",
    func= mean_info_pro, #meaningfull information provider 
    description=""
)

tools = [retrieval, db_mongo]


# Master Agent that works like the CEO and give commands to other agents
Master_agent = create_openai_tools_agent(llm=llm,tools=tools,prompt=prompt)
conversation = AgentExecutor(agent=Master_agent,tools=tools,memory=memory,verbose=True)


def user_input(input : str):
    user_input = input

    if user_input:
        response =conversation.invoke(input={"input": user_input})  # Provide me the car of this customer id CUST-2FE819C3
        print("FIneal Resposne \n\n:-",response['output'])

        return response['output']






# print("\nTop Retrieved Documents:")
# for i, doc in enumerate(retrieved_docs, 1):+
#     print(f"{i}. {doc.page_content[:400]}...") 
# print("Length of the Documents:- ",len(documents))