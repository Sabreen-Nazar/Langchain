
import os

from constants import openai_key
from langchain.llms import OpenAI
import streamlit as st
from langchain import PromptTemplate
from langchain.chains import LLMChain  # this LLM chain is needed whenever you use PropmtTemplate
from langchain.chains import SequentialChain # to combine chain 1 , 2 etc.,
from langchain.memory import ConversationBufferMemory  # to save the Memory 
#llm = OpenAI(openai_api_key = openai_key)
os.environ["OPENAI_API_KEY"] = openai_key

# Initialise streamlit framework


st.title("Explore any Topic")
input_text = st.text_input("Search the topic")

# promt template  

first_promt = PromptTemplate(
    input_variables = ['topic'] , 
    template= "Get in depth document about {topic}" 
)
# for every promt template we need LLM chain
# LLM 
llm = OpenAI(temperature=0.9)   
# Memory 

doc_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
ques_memory = ConversationBufferMemory(input_key='details', memory_key='chat_history')
web_memory = ConversationBufferMemory(input_key='ques', memory_key='description_history')

chain1 = LLMChain(llm = llm , prompt = first_promt , verbose = True ,output_key = 'details') # this output key is used in mext chain 


second_promt = PromptTemplate(
    input_variables = ['details'] , 
    template= "form some atleast 10 questions and answers from the document  {details} for Senior Position Interview"   
)
chain2 = LLMChain(llm = llm , prompt = second_promt , verbose = True ,output_key = 'ques')

third_promt = PromptTemplate(
    input_variables = ['ques'] , 
    template= "Say some websites for studying {ques}"   
)
#llm = OpenAI(temperature=0.9)   
chain3 = LLMChain(llm = llm , prompt = third_promt , verbose = True ,output_key = 'web')

chain = SequentialChain(chains = [chain1 , chain2 , chain3]  , 
                        input_variables=['topic'] ,
                        output_variables= ['details','ques','web'],
                        verbose= True)

# if input_text:
#     st.write(llm(input_text))


# print output
if input_text:
    st.write(chain({'topic':input_text}))

    with st.expander('Doc'): 
        st.info(doc_memory.buffer)

    with st.expander('Q_A'): 
        st.info(ques_memory.buffer)
#llm.predict(" What is Lanchain - Explain")




