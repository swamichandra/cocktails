"""Python file to serve as the frontend"""
import random
import os
import css
import re
import openai
import streamlit as st
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import SimpleSequentialChain
import datetime
import json
import s3fs
import boto3

# All of Streamlit config and customization
st.set_page_config(page_title="Cocktail Maker powered by Generative AI", page_icon=":random:", layout="wide")
st.write(f'<style>{css.v1}</style>', unsafe_allow_html=True)
ss = st.session_state
if 'debug' not in ss:
    ss['debug'] = {}
st.markdown(""" <style>
#MainMenu {visibility: visible;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

# Remove whitespace from the top of the page and sidebar
st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 1rem;
                    padding-right: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)

#START LLM portions 
if os.getenv("OPENAI_API_KEY") is not None:
    pass
else:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

if os.environ["OPENAI_API_KEY"]:
    #st.image('logo3.png')
    st.title("🅼🅰️🆂🆃🅴🆁 🅼🅸🆇🅾🅻🅾🅶🅸🆂🆃")
    #st.title("🅜🅰️🅢🅣🅔🅡 🅜ℹ🅧🅞🅛🅞🅖🅘🅢🅣")
    st.caption("Let generative AI come up with new drink recipes for you")
else:
    st.error("🔑 Please enter API Key")

cocktail_name = ""
citations = ""

# Available Models
PRIMARY_MODEL = 'gpt-3.5-turbo' #'gpt-4-0314' ' #"text-davinci-002"

#This is an LLMChain to generate a cocktail and associated instructions.

# Number between -2.0 and 2.0. 
# Positive values penalize new tokens based on their existing frequency in the text so far, 
# decreasing the model's likelihood to repeat the same line verbatim.
FREQ_PENALTY = 0

# Number between -2.0 and 2.0. 
# Positive values penalize new tokens based on whether they appear in the text so far, 
# increasing the model's likelihood to talk about new topics
PRESENCE_PENALTY = 0

llm = ChatOpenAI(model_name=PRIMARY_MODEL, temperature=0.15, top_p=1, frequency_penalty=FREQ_PENALTY, presence_penalty=PRESENCE_PENALTY, max_tokens=600)

template = """You are a talented mixologist skilled at crafting palate-pleasing drinks. Please suggest  an aroma-friendly and olfactory pleasant {drink} to complement {main_dish} at a {occasion}. The recipe should include {ingredient}, pair well with the dish's cuisine, and use affordable, common ingredients. Please avoid meat, eggs, and yolk. Create a trendy four-word drink name, and provide additional details as requested. {additional_instructions}.

Cocktail Name:

Ingredients:

Instructions:

Rationale:

Olfactory Flavor Theory:

Pairing with Main Dish:

Considered but Excluded Drink Pairing:

Catch Phrase:

Citations & Inspiration for the Recipe:

Shopping List:
"""

prompt_4_cocktail = PromptTemplate(input_variables=["drink", "ingredient", "occasion", "additional_instructions", "main_dish"], template=template.strip(),)
cocktail_gen_chain = LLMChain(llm=llm, prompt=prompt_4_cocktail, output_key="cocktail", verbose=True)

#This is an LLMChain to generate a short haiku caption for the cocktail based on the ingredients.
llm = OpenAI(model_name=PRIMARY_MODEL, temperature=0.2, top_p=1, frequency_penalty=FREQ_PENALTY, presence_penalty=PRESENCE_PENALTY, max_tokens=75)

#template2 = """Write a restaurant menu style description for a {drink} that has the following ingredients {ingredient}, suitable for a {occasion} occasion. It must pair well with {main_dish}. Strictly 50 words only. Only generate complete sentences. Be crisp and short."""

template2 = "Create a 50-word, restaurant menu description for a {drink} featuring {ingredient}, ideal for a {occasion} event. Ensure it complements {main_dish}. Use complete sentences and concise language for fast, high-quality results."

prompt_4_caption = PromptTemplate(input_variables=["drink", "ingredient", "main_dish", "occasion"], template=template2.strip(),)
cocktail_caption_chain = LLMChain(llm=llm, prompt=prompt_4_caption, output_key="caption", verbose=True)

#This is the overall chain where we run these two chains in sequence.
overall_chain = SequentialChain(
    chains=[cocktail_gen_chain, cocktail_caption_chain],
    input_variables=['drink', 'ingredient', 'occasion', 'additional_instructions', 'main_dish'],
    # Here we return multiple variables
    output_variables=['cocktail', 'caption'],
    verbose=True)
#END LLM portions


# From here down is all the StreamLit UI.
occasion_list = ["Family Celebration", "Birthday", "Team Event", "Wedding", "Winter Gathering", "Retirement", "Valentine’s Day", "Holiday Celebration", "Summer Picnic", "Labor Day", "All Occasions"]
occasion_list = sorted(occasion_list)


ingredients = ['Bourbon', 'Brandy', 'Gin', 'Grappa', 'Pisco', 'Port', 'Rum', 'Sherry', 'Single Malt Scotch', 'Tequila', 'Vermouth', 'Vodka', 'Whisky', 'Wine', 'Agave', 'Apple Slice', 'Lemon Twist', 'Mango Pulp', 'Mint Leaves', 'Orange Slice', 'Club Soda', 'Coffee Concentrate', 'Coke', 'Root Beer', 'Honey']

ingredients_nonalcoholic = ['Agave', 'Apple', 'Banana', 'Blackberries', 'Blueberries', 'Buttermilk', 'Club Soda', 'Cocktail Umbrellas', 'Coffee Concentrate', 'Coke', 'Edible Flowers', 'Grapefruit Juice', 'Honey Syrup', 'Lassi', 'Lavender', 'Lemon', 'Lemon Juice', 'Lemon and Lime Zest', 'Lime Juice', 'Lyre American Malt', 'Mango Lassi', 'Maple Syrup', 'Maraschino Cherry', 'Mint Leaves', 'Orange', 'Orange Juice', 'Peach', 'Pear', 'Pepsi', 'Pineapple', 'Pineapple Juice', 'Raspberries', 'Ritual Gin', 'Ritual Tequila', 'Rosemary', 'Sage', 'Salt Lassi', 'Seedlip', 'Strawberries', 'Thyme', 'Tonic Water', 'Yogurt']

NON_ALCOHOLIC_FLAG = False
drink = ''
main_dish = 'all dishes'

def get_ingredient():
    if(drink == 'Non-Alcoholic'):
        input_text = ', '.join(random.choices(ingredients_nonalcoholic, k=3)) + ", " + ', '.join(optional_ingredient)
    else:
        input_text = ', '.join(random.choices(ingredients, k=1)) + ", " + ', '.join(optional_ingredient)
    return input_text

optional_ingredient = ""

#You can check .empty documentation
placeholder = st.empty()

with placeholder.container():
    with st.form('app'):
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        with col1:
            drink = st.selectbox('Type of drink', options=['Cocktail', 'Shot', 'Punch', 'Non-Alcoholic'])
            #print(drink)
            if(drink == 'Non-Alcoholic'):
                NON_ALCOHOLIC_FLAG = True
                #drink = 'No Alcohol Mocktail'
                
        with col2:
            occasion = st.selectbox('Optionally, what is the occasion?', options=occasion_list)
            
        with col3:
            if NON_ALCOHOLIC_FLAG:
                optional_ingredient = st.multiselect(label='Optionally, any particular ingredients?', options=ingredients_nonalcoholic,)
            else:
                optional_ingredient = st.multiselect(label='Optionally, any particular ingredients?', options=ingredients,)
            #print(optional_ingredient)

        with col4:
            main_dish = st.text_input("Optionally, a dish to pair with")

        btn = st.form_submit_button("GENERATE")

    if btn:
        ingredient_input = get_ingredient()
        
        #print(">>>>", len(main_dish))
        if len(main_dish) <= 0:
            main_dish = 'all dishes'
        
        #print("*******LLM Prompt")
        #print(prompt_4_cocktail)
        
        with st.spinner(text="Building your "  + drink + " recipe ... that pairs well with  " + main_dish + " for your " + occasion + " occasion."):
            if NON_ALCOHOLIC_FLAG:
                output = overall_chain({'drink': drink, 'ingredient': ingredient_input, 'occasion': occasion, 'cocktail_name': cocktail_name, 'additional_instructions':'Do not include any alcohol. No whisky, cognac, spirits, VSOP, wine, bourbon, gin, scotch, beer in the ingredients', 'main_dish': main_dish})
            else:
                output = overall_chain({'drink': drink, 'ingredient': ingredient_input, 'occasion': occasion, 'cocktail_name': cocktail_name, 'additional_instructions': '', 'main_dish': main_dish})
            print("*******")
            print(output)
            print("*******")
            cocktail_name = output['cocktail'][:output['cocktail'].index("Ingredients")]
            cocktail_name = cocktail_name.strip().partition("Cocktail Name:")[2].strip()
            output['cocktail_name'] = cocktail_name
            #st.header(cocktail_name)
            #print(cocktail_name)
            
            st.header(cocktail_name)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("How to mix this?")
                ingredients_list = output['cocktail'].strip().partition("Ingredients:")[2]
                print(output['cocktail'].strip().partition("Ingredients:")[2])
                ingredients_list = output['cocktail'][:output['cocktail'].index("Rationale")]
                st.markdown(ingredients_list)

            with col2:
                st.subheader("How will this drink look?")
                #st.markdown(drink)
                
                print("*******Diffusion Prompt")
                prompt_4_diffusion = "Realistic studio photo of a drink of type " + drink + "  and named as " + cocktail_name + ". Apt for " + occasion + " - chrome backlighting, another dimension, digital art"
                
                #prompt_4_diffusion = drink + " drink named " + cocktail_name + ". Apt for " + occasion + ". Magazine cover. No human images or faces please --ar 4:3 --v 4 --c 100"
                #st.markdown(prompt_4_diffusion.strip())
                #st.button("📷 Share")# take screenshot using pyautogui
                #image = pyautogui.screenshot()
                #image.show()  # Show the image using the default image viewer
                print(prompt_4_diffusion)

                kwargs = {
                    "prompt": prompt_4_diffusion,
                    "n": 1,
                    "size": '512x512'}
                image_resp = openai.Image.create(**kwargs)
                output['diffusion_image'] = image_resp
                #st.code(image_resp)
                image_url = image_resp['data'][0]['url']
                st.image(image_url)
                st.caption(output['caption'].strip())

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("How did I come up with this?")
                
                try:
                    
                    start_index = output['cocktail'].strip().partition("Rationale:")[2].find("Shopping List:")
                    if start_index != -1:
                        st.markdown(output['cocktail'].strip().partition("Rationale:")[2][:start_index])
                    else:
                        st.markdown(output['cocktail'].strip().partition("Rationale:")[2])
                    
                    #st.markdown(output['cocktail'].strip().partition("Rationale:")[2])
                    #print("****___****")
                    #print(output['cocktail'].strip().partition("Rationale:")[2])
                    #print("****___****")
                except ValueError:
                    st.markdown("")
                    
                try:
                    st.subheader("Shopping List ")
                    st.caption("Here is a shopping list to prepare the drink")
                    slist = output['cocktail'].strip().partition("Rationale:")[2]
                    slist = slist.strip().partition("Shopping List:")[2]
                    
                    #ingredients = slist
                    #ilist = slist.split("\n")
                    #print(ilist)

                    #for i in ilist:
                    #    href = f"<a href='{i}'>{i}</a>"
                    #    st.markdown(href)
                    
                    st.markdown(slist)
                except ValueError:
                    st.markdown("")
                    
            
            with col2:
                st.subheader("Under the Covers: Goal, Plan & Chain ")
                st.markdown(f"""###### Goal: Come up with an olfactory pleasant {drink}""")
                st.markdown(f"""**Plan**: Draw Inspiration ➺ Pick Ingredients ➺ Generate Drink ➺ Mixing Instructions ➺ Visualize Drink ➺ Create Shopping List ➺ Provide Explanation""")
                st.json(output)
                
            # Create an S3 Client 
            s3_client = boto3.client('s3') 

            # Data to be written to S3 Bucket 
            # Convert Data to JSON 
            json_data = json.dumps(output)
            
            s3_key = cocktail_name.replace(" ", "_") + '_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Write JSON to S3 Bucket 
            s3_client.put_object(Body = json_data, Bucket = 'mixologist', Key = str(s3_key) + '.json', ContentType = 'application/json')

st.caption("Non-Humanoid Developer: Swami Chandrasekaran")
