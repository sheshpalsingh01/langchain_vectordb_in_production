from langchain_openai import OpenAI
from langchain_classic.chains import ConversationChain  
from langchain_classic.memory import ConversationBufferMemory  

llm = OpenAI(model="gpt-4o-mini", temperature=0)

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)


# Start the conversation
conversation.predict(input="Tell me about yourself.")
conversation.predict(input="What can you do?")
conversation.predict(input="How can you help me with data analysis?")

# Display the conversation history
print(conversation.memory.buffer)