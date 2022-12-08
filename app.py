# Import necessary libraries
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Define a function to calculate the semantic similarity between two pieces of text
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
def semantic_similarity(text1,text2):
  t1 = model.encode(text1)
  t2 = model.encode(text2)
  similarity = cosine_similarity([t1], [t2])[0][0]
  return similarity * 100
# Create the Streamlit app
st.title("Semantic Similarity Calculator")

# Get the input text from the user
text1 = st.text_input("Enter the first piece of text:")
text2 = st.text_area("Enter the options separated by pipe symbol (|):",height=50)

# Ask the user to check a checkbox before running the model
run_model = st.button("Run model")

# If the user has checked the checkbox, calculate the semantic similarity
# between the two pieces of text and display the result
if run_model:
  text_2 = text2.split("|")
  sim_list = []
  for txt in text_2:
    similarity = semantic_similarity(text1, txt)
    sim_list.append(similarity)
  max_index = sim_list.index(max(sim_list))
  st.write("Similarity: %.2f%%" % similarity)
  st.write("Recommendation: ",text_2[max_index])
  d = {}
  for i in range(len(text_2)):
    d[text_2[i]] = "Score : %.2f%%" % sim_list[i]
  st.write("Following are the option wise scores")
  st.write(d)