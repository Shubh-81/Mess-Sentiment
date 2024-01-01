import subprocess

subprocess.run(["pip", "install", "langchain"])

from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import FewShotPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
import streamlit as st

llm = Ollama(model='llama2', callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

examples = [
    {"phrase": "The mess food is delicious, and I look forward to every meal.", "class": "Alpha"},
    {"phrase": "I find the mess food to be quite bland and uninspiring.", "class": "Beta"},
    {"phrase": "The mess staff is attentive, and they cater to dietary preferences.", "class": "Alpha"},
    {"phrase": "I often skip meals at the mess because the food quality is inconsistent.", "class": "Beta"},
    {"phrase": "The variety of options in the mess menu is impressive.", "class": "Alpha"},
    {"phrase": "I'm disappointed with the lack of vegetarian options in the mess.", "class": "Beta"},
    {"phrase": "The mess provides a convenient and affordable dining option.", "class": "Alpha"},
    {"phrase": "The mess food is not worth the money; I'd rather eat elsewhere.", "class": "Beta"},
    {"phrase": "I appreciate the effort the mess puts into themed dinners.", "class": "Alpha"},
    {"phrase": "The mess cleanliness standards need improvement.", "class": "Beta"},
    {"phrase": "The mess desserts are a delightful way to end the meal.", "class": "Alpha"},
    {"phrase": "The mess food makes me feel sick; I avoid it whenever I can.", "class": "Beta"},
    {"phrase": "The mess accommodates special dietary needs well.", "class": "Alpha"},
    {"phrase": "The mess food is repetitive, and I get tired of eating the same things.", "class": "Beta"},
    {"phrase": "The mess atmosphere is lively and encourages socializing.", "class": "Alpha"},
    {"phrase": "The mess food lacks freshness; it tastes like leftovers.", "class": "Beta"},
    {"phrase": "The mess timings are convenient for my schedule.", "class": "Alpha"},
    {"phrase": "I've had multiple instances of finding foreign objects in the mess food.", "class": "Beta"},
    {"phrase": "The mess food caters to a diverse range of tastes.", "class": "Alpha"},
    {"phrase": "The mess portion sizes are too small; I'm often left hungry.", "class": "Beta"},
    {"phrase": "I enjoy the cultural diversity reflected in the mess cuisine.", "class": "Alpha"},
    {"phrase": "The mess food is overpriced for its quality.", "class": "Beta"},
    {"phrase": "The mess hygiene standards are commendable.", "class": "Alpha"},
    {"phrase": "The mess food lacks nutritional value.", "class": "Beta"},
    {"phrase": "The mess offers a good balance between healthy and indulgent options.", "class": "Alpha"},
    {"phrase": "I can't stand the taste of the mess food; it's unbearable.", "class": "Beta"},
    {"phrase": "The mess takes feedback seriously and makes improvements.", "class": "Alpha"},
    {"phrase": "The mess service is slow, and I don't have time to wait.", "class": "Beta"},
    {"phrase": "The mess caters to different dietary preferences, including jain options.", "class": "Alpha"},
    {"phrase": "The mess food is too oily; it affects my health negatively.", "class": "Beta"},
]

example_template = """
{{phrase}}
Class: {{class}}
"""

prompt = PromptTemplate.from_template(example_template, template_format='jinja2')

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=prompt,
    prefix="Few-shot classification, only provide one word answer, no explanation needed.",
    suffix="Find the class of this phrase: {input}, based on the previous examples.",
    input_variables=["input"],
    example_separator="\n\n",
)

chain = LLMChain(llm=llm, prompt=few_shot_prompt)

st.title("Mess Review Sentiment Analyzer")

st.markdown("Enter a review and click the 'Analyze' button to find its sentiment.")
user_input = st.text_input("Enter a mess review:")

classify_button = st.button("Analyze", key="classify_button")

st.write("")

if classify_button:
    result = chain.predict(input=user_input)
    if "alpha" in result.lower():
        st.write(f"**Sentiment:** Positive", key="result_output")
    else:
        st.write(f"**Sentiment:** Negative", key="result_output")
