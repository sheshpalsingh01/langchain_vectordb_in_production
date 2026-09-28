import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
# Use ChatOpenAI for chat/instruct models
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

# LCEL chain: prompt | llm | output parser
chain = prompt | llm | StrOutputParser()

# Invoke with a dictionary
print(chain.invoke({"product": "eco-friendly water bottles"}))
