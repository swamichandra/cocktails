"""Python file to serve as the frontend"""
import random
import os
import re
import openai
import streamlit as st
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.chains import SimpleSequentialChain
from PIL import Image
from PIL import ImageGrab

# All of Streamlit config and customization
st.set_page_config(page_title="Cocktail Maker powered by Generative AI", page_icon=":random:", layout="wide")
st.markdown(""" <style>
 div.stButton > button {{
        
        display:inline-block;outline:0;cursor:pointer;border:none;padding:0 56px;height:45px;line-height:45px;border-radius:7px;background-color:#5643CC;width:100%;color:white;font-weight:400;font-size:16px;box-shadow:0 4px 14px 0 rgb(0 118 255 / 39%);transition:background 0.2s ease,color 0.2s ease,box-shadow 0.2s ease;:hover{{background:rgba(0,118,255,0.9);box-shadow:0 6px 20px rgb(0 118 255 / 3%)}}
        
}}
#MainMenu {visibility: visible;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)
padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

st.markdown(
    """
<style>
.st-au {{
  font-size: .5rem;
}}
</style>
""",
    unsafe_allow_html=True,
)



with open( "style.css" ) as css: st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

#START LLM portions

if os.environ["OPENAI_API_KEY"]:
    st.title("Master Mixologist")
    st.caption("Let generative AI come up with new drink recipes for you")
else:
    st.error("🔑 Please enter API Key")

cocktail_name = ""

# Available Models
LANGUAGE_MODELS = ['gpt-3.5-turbo', 'text-davinci-003', 'text-curie-001', 'text-babbage-001', 'text-ada-001']
CODEX_MODELS = ['code-davinci-002', 'code-cushman-001']
PRIMARY_MODEL = 'gpt-3.5-turbo' #'gpt-4-0314' ' #"text-davinci-002"
SECONDARY_MODEL = 'text-curie-001'

#This is an LLMChain to generate a cocktail and associated instructions.

# Number between -2.0 and 2.0. 
# Positive values penalize new tokens based on their existing frequency in the text so far, 
# decreasing the model's likelihood to repeat the same line verbatim.
FREQ_PENALTY = 1.02

# Number between -2.0 and 2.0. 
# Positive values penalize new tokens based on whether they appear in the text so far, 
# increasing the model's likelihood to talk about new topics
PRESENCE_PENALTY = 1.02

llm = OpenAI(model_name=PRIMARY_MODEL, temperature=1, frequency_penalty=FREQ_PENALTY, presence_penalty=PRESENCE_PENALTY, max_tokens=600, top_p=1)
#llm = OpenAIChat(model_name=PRIMARY_MODEL, temperature=1, frequency_penalty=FREQ_PENALTY, presence_penalty=PRESENCE_PENALTY, max_tokens=600, top_p=1)

template = """I want someone who can suggest out of the world and imaginative drink recipes. You are my master mixologist. You will come up with olfactory pleasant {drink} that is appealing and pairs well with the {cuisine} cuisine. Use {ingredient} in your recipe. Draw inspiration from an existing cocktail recipe of {inspiration}. Apply understanding of flavor compounds and food pairing theories. Give the drink a unique name. Ingredients must start in a new line. Add a catch phrase for the drink within double quotes. Provide a scientific explanation for why the ingredients were chosen. Do not include eggs and yolk as ingredients.
Cocktail Name: 
Ingredients:
Instructions:
Rationale:###
"""

prompt_4_cocktail = PromptTemplate(input_variables=["drink", "ingredient", "inspiration", "cuisine"],template=template.strip(),)
cocktail_gen_chain = LLMChain(llm=llm, prompt=prompt_4_cocktail, output_key="cocktail", verbose=True)

#This is an LLMChain to generate a short haiku poem for the cocktail based on the ingredients.
llm = OpenAI(model_name=SECONDARY_MODEL, temperature=0.7, frequency_penalty=FREQ_PENALTY, presence_penalty=PRESENCE_PENALTY, max_tokens=200, best_of=3, top_p=0.5)

template2 = """###Write a restaurant menu style short description for a {drink} that has the following ingredients {ingredient} and pairs well with {cuisine} cuisine###."""

prompt_4_poem = PromptTemplate(input_variables=["drink", "ingredient", "cuisine"], template=template2.strip(),)
cocktail_poem_chain = LLMChain(llm=llm, prompt=prompt_4_poem, output_key="poem", verbose=True)

#This is the overall chain where we run these two chains in sequence.
overall_chain = SequentialChain(
    chains=[cocktail_gen_chain, cocktail_poem_chain],
    input_variables=['drink', 'ingredient', 'inspiration', 'cuisine'],
    # Here we return multiple variables
    output_variables=['cocktail', 'poem'],
    verbose=True)
#END LLM portions


# From here down is all the StreamLit UI.
ingredients = ['42 Below Kiwi Vodka', '42 Below Manuka Honey Vodka', '9th Street Alphabet City Coffee Concentrate', 'A.B. Smeby Verbena Bitters', 'Abbotts Bitters', 'Adam Elmegirabs Bokers Bitters', 'Afel Anise Hyssop Essence', 'Aftel Clove Essence', 'Aftel Tobacco Essence', 'Agave', 'Agave Nectar', 'Agave Syrup', 'Al Wadi Pomegranate Molasses', 'Amaro Ciociaro', 'Amass', 'Amer Picon', 'Anchor Genevieve', 'Angostura Bitters', 'Angostura Orange Bitters', 'Aperol', 'Apple', 'Apple Fan', 'Apple Slice', 'Appleton Estate Reserve Rum', 'Appleton Estate V/X Rum', 'Ardbeg 10-Year-Old Single Malt Scotch Whisky', 'Asparagus Tip', 'Averna Amaro', 'Aviation Gin', 'Bacardi 8 Rum', 'Banana', 'Banks 5 Island Rum', 'Barbancourt 8-Year-Old Rhum', 'Barbancourt Rhum Blanc', 'Barenjager Honey Liqueur', 'Barsol Quebranta Pisco', 'Beaujolais Nouveau', 'Beefeater 24 Gin', 'Beefeater Gin', 'Bek Se Ju 100-year wine', 'Beleza Pura Cachaca', 'Belle de Brillet', 'Belvedere Vodka', 'Benedictine', 'Benromach 12-Year-Old Single Malt Scotch Whisky', 'Bentons Bacon Fat-Infused Four Roses Bourbon', 'Berkshire Mountain Distillers Greylock Gin', 'Birch-Infused Rittenhouse Bonded Rye Whiskey', 'Bittermens Xocolatl Mole Bitters', 'Black Bush Irish Whiskey', 'Black Cardamom Syrup', 'Black Sesame-Infused Krogstad Aquavit', 'BlackTea-Infused Elijah Craig 12-Year-Old Bourbon', 'Blackberries', 'Blackberry', 'Blandys Sercial Madeira', 'Blood Orange Juice', 'Blueberries', 'Blueberry', 'Boiron Passion Fruit Puree', 'Boiron Rhubarb Puree', 'Bols Genever', 'Bonne Maman Apricot Preserves', 'Bonne Maman Orange Marmalade', 'Bonne Maman Raspberry Preserves', 'Bonne Maman Strawberry Preserves', 'Bookers Bourbon', 'Borsci Sambuca', 'Brandied Cherry', 'Brooklyn Black Chocolate Stout', 'Brooklyn Brewery Local 1', 'Bulleit Bourbon', 'Bushmills Irish Whiskey', 'Buttermilk', 'Campari', 'Candied Ginger', 'Canton Ginger Liqueur', 'Caramelized Simple Syrup', 'Carlshamns Flaggpunch', 'Carpano Antica Sweet Vermouth', 'Carpene Malvolti Prosecco', 'Cayenne', 'Celery Salt', 'Celery Stalk', 'Chamomile-Infused Barsol Quebranta Pisco', 'Chamomile-Infused Compass Box Asyla Blended Scotch Whisky', 'Channing Daughters Scuttlehole Chardonnay', 'Cherry', 'Cherry Heering', 'Chilled Brewed Hibiscus Tea', 'Chivas Regal 12-Year-Old Blended Scotch Whiskey', 'Cholula', 'Ciaco Bella Coconut Sorbet', 'Cinnamon Stick', 'Clear Creek Kirschwasser', 'Clear Creek Pear Brandy', 'Clear Creek Plum Brandy', 'Clove', 'Clove Syrup', 'Club Soda', 'Coca-Cola Classic', 'Cocchi Americano', 'Cocktail Umbrellas', 'Cocoa Powder', 'Cointreau', 'Compass Box Asyla Blended Scotch Whisky', 'Compass Box Oak Cross Blended Malt Scotch Whisky', 'Compass Box Peat Monster Blended Malt Scotch Whisky', 'Concord Grape', 'Concord Shrubb', 'Corn Water', 'Cranberry Syrup', 'Creme Yvette', 'Cruzan Black Strap Rum', 'Cucumber Ribbon', 'Cucumber Slice', 'Cucumber Wheel', 'Cynar', 'Dandelion Root-Infused Rittenhouse Bonded Rye Whiskey', 'Deep Mountain Grade B Maple Syrup', 'Dehydrated Citrus', 'Del Maguey Vida Mezcal', 'Demerara Sugar', 'Demerara Sugar Cube', 'Demerara Sugar cube soaked in Angostura Bitters', 'Demerara Syrup', 'Dill Sprig', 'Diluted Aftel Bergamot Essence', 'Diluted Aftel Black Pepper Essence', 'Dolin Blanc Vermouth', 'Dolin Dry Vermouth', 'Dolin Sweet Vermouth', 'Don Julio Anejo Tequila', 'Don Julio Reposado Tequila', 'Dows Ruby Port', 'Dows Tawny Port', 'Dr. Konstantin Frank Dry Riesling', 'Drambuie', 'Dried Lavender Sprig', 'Dried Persimmon Slice', 'Drouhin Pommeau', 'Dubonnet Rouge', 'Dupont Brut Sparkling Cider', 'Edible Flowers', 'Edible Orchid', 'Edouard Absinthe', 'Egg White', 'Egg Yolk', 'El Dorado 15-Year-Old Rum', 'El Tesoro Anejo Tequila', 'El Tesoro Platinum Tequila', 'El Tesoro Reposado Tequila', 'Elijah Craig 12-Year-Old Bourbon', 'Eurovanille Vanilla Syrup', 'Famous Grouse Blended Scotch Whisky', 'Fee Brothers Grapefruit Bitters', 'Fee Brothers Old Fashion Bitters', 'Fee Brothers Peach Bitters', 'Fee Brothers Rhubarb Bitters', 'Fee Brothers Whiskey Barrel Aged Bitters', 'Feldmans Barrel Aged Bitters', 'Fennel Bulb Slice', 'Fernet Branca', 'Fever-Tree Bitter Lemon Soda', 'Fever-Tree Ginger Ale', 'Fitzs Root Beer', 'Flamed Orange Twist', 'Flor de Cana Silver Dry Rum', 'Freshly Brewed Chamomile Tea', 'Freshly Brewed Coffee', 'Freshly Peeled Ginger Slice', 'Freshly Whipped Cream', 'Galliano LAutentico', 'George Dickel No. 12 Tennessee Whisky', 'George T. Stagg Bourbon', 'Glen Thunder Corn Whiskey', 'Glenlivet 12-Year-Old Single Malt Scotch Whisky', 'Godiva Original Liqueur', 'Goji Berry-Infused Four Roses Single Barrel Bourbon', 'Golden Star Sparkling White Jasmine Tea', 'Goslings Black Seal Rum', 'Gran Centenario Blanco Tequila', 'Gran Centenario Reposado Tequila', 'Gran Classico Bitter', 'Gran Duque DAlba Brandy de Jerez', 'Grand Marnier', 'Granny Smith Apple Slice', 'Grapefruit Juice', 'Grapefruit Syrup', 'Grapefruit Twist', 'Green Chartreuse', 'Green Chartreuse V.E.P.', 'Green Pepper Slice', 'Ground Black Pepper', 'Guldens Spicy Brown Mustard', 'Half a Grapefruit Wheel', 'Half an Orange Wheel', 'Hangar One Buddhas Hand Vodka', 'Havana Club 7-Year-Old Rum', 'Haymans Old Tom Gin', 'Heart of the Hudson Apple Vodka', 'Heavy Cream', 'Hendricks Gin', 'Herb Pharm Goldenseal Tincture', 'Hibiscus-Infused Bernheim Wheat Whiskey', 'Hine V.S.O.P. Cognac', 'Honey Nut Cheerios', 'Honey Syrup', 'Honeydew Melon Ball', 'Honeydew Melon Juice', 'Horchata', 'Hot Water', 'House Ginger Beer', 'House Grenadine', 'House Orange Bitters', 'Ice-Cold Filtered Water', 'Illegal Reposado Mezcal', 'Illy Espresso Liqueur', 'J.M. Rhum Blanc', 'Jalapeno Slice no seeds', 'Jalapeno Slice with few seeds', 'Jameson 12-Year-Old Irish Whiskey', 'Jameson Irish Whiskey', 'John D. Taylors Velvet Falernum', 'Jose Cuervo Platino Tequila', 'Jose Cuervo Tradicional Tequila', 'Jujube Tea-infused Vya Sweet Vermouth', 'Kahlua', 'Kamoizumi Nigori Sake', 'Kamoizumi Shusen Sake', 'Karlssons Gold Vodka', 'Kassatly Chtaura Orgeat', 'Kirsch Brandied Cherry', 'Kosher Salt', 'Krogstad Aquavit', 'Kubler Absinthe', 'Kumquat Syrup', 'L & J Blanco Tequila', 'La Diablada Pisco', 'Lairds Applejack', 'Lairds Bonded Apple Brandy', 'Lakewood Cranberry Juice', 'Laphroaig 10-Year-Old Single Malt Scotch', 'Large Straw', 'Lavender', 'Lavender Tincture', 'Lemon', 'Lemon Balm', 'Lemon Hart Overproof Rum', 'Lemon Juice', 'Lemon Peel', 'Lemon Syrup', 'Lemon Twist', 'Lemon Wedge', 'Lemon Wheel', 'Lemon and Lime Zest', 'Lemongrass Syrup', 'Libbys Pumpkin Puree', 'Licor 43', 'Lillet Blanc', 'Lillet Rouge', 'Lime Cordial', 'Lime Disc', 'Lime Juice', 'Lime Twist', 'Lime Wedge', 'Lime Wheel', 'Lime Zest', 'Linie Aquavit', 'Liquiteria Coconut Water', 'Lustau Cream Sherry', 'Lustau East India Sherry', 'Lustau Manzanilla Sherry', 'Lustau Palo Cortado Sherry', 'Lustau Pedro Ximenez Sherry', 'Luxardo Amaretto', 'Luxardo Bitter', 'Luxardo Maraschino Liqueur', 'Lyre American Malt', 'Macchu Pisco', 'Macerated Cranberry', 'Mae de Ouro Cachaca', 'Makers Mark Bourbon', 'Mandarin Napoleon', 'Mape Syrup', 'Maraschino Cherry', 'Maraska Maraschino Liqueur', 'Marie Brizard Creme de Banane', 'Marie Brizard Dark Creme de Cacao', 'Marie Brizard Orange Curacao', 'Marie Brizard White Creme de Cacao', 'Marivani Lavender Essence', 'Marivani Orange Flower Water', 'Marivani Rose Flower Water', 'Martell V.S.O.P. Cognac', 'Martini Bianco Vermouth', 'Martini Sweet Vermouth', 'Martinique Sugar Cane Syrup', 'Masumi Arabashiri Sake', 'Masumi Okuden Junmai Sake', 'Mathilde Pear Liqueur', 'Mathilde Peche', 'Matusalem Gran Reserva Rum', 'Mint Leaf', 'Mint Leaves', 'Mint Sprig', 'Moet Imperial Champagne', 'Monteverdi Nocino', 'Mount Gay Eclipse Amber Rum', 'Mount Gay Eclipse White Rum', 'Mount Gay X.O. Rum', 'Myerss Dark Rum', 'Mymoune Rose Syrup', 'Navan Vanilla Liqueur', 'Neisson Rhum Blanc', 'Neisson Rhum Reserve Speciale', 'Nikka Taketsuru 12-Year-Old Japanese Malt Whisky', 'Noilly Prat Dry Vermouth', 'Nonino Amaro', 'Nonino Gioiello', 'Noval Black Port', 'Oban 14-Year-Old Single Malt Scotch Whisky', 'Ocho Anejo Tequila', 'Old Grand-Dad Bonded Bourbon', 'Old Overholt Rye Whiskey', 'Old Potrero Hotalings Rye Whiskey', 'Orange', 'Orange Juice', 'Orange Peel', 'Orange Slice', 'Orange Twist', 'Orange Wedge', 'Orange Wheel', 'Orange Zest', 'Orange-Cherry Flag', 'Oregano Sprig', 'Oud Beersel Framboise', 'Pama Pomegranate Liqueur', 'Pampero Aniversario Rum', 'Pansy Flower', 'Partida Blanco Tequila', 'Partida Reposado Tequila', 'Paumanok Cabernet Franc', 'Peach', 'Pear', 'Pepsi', 'Perfect Purees of Napa Valley Prickly Pear Puree', 'Pernod', 'Peruvian Amargo Bitters', 'Peychauds Bitters', 'Pickled Ramp Brine', 'Pickled Ramps', 'Pierre Ferrand Ambre Cognac', 'Pimms No. 1 Cup', 'Pinch Grated Cinnamon', 'Pinch Grated Nutmeg', 'Pinch Ground Chili', 'Pinch Kosher Salt', 'Pinch Sea Salt', 'Pineapple', 'Pineapple Juice', 'Pineapple Leaf', 'Pineapple Slice', 'Pink Rose Petal', 'Pitted Cherry', 'Plymouth Gin', 'Plymouth Sloe Gin', 'Popcorn-Infused Flor de Cana Silver Dry Rum', 'Punt e Mes', 'Q Tonic', 'Quince Shrubb (Huilerie Beaujolaise Vinaigre de Coing)', 'Raisins', 'Ransom Old Tom Gin', 'Raspberries', 'Red Bell Pepper Slice', 'Red Jacket Orchards Apple Butter', 'Red Jacket Orchards Apple Cider', 'Regans Orange Bitters', 'Remy Martin V.S.O.P. Cognac', 'Rhum Clement Creole Shrubb', 'Rhum Clement V.S.O.P.', 'Ricard Pastis', 'Rittenhouse Bonded Rye Whiskey', 'Ritual Gin', 'Ritual Tequila', 'Rock Candy Swizzle', 'Ron Zacapa 23 Centenario Rum', 'Ron Zacapa Centenario 23 Rum', 'Rose-Infused Plymouth Gin', 'Rosemary', 'Rosemary Sprig', 'Rothman & Winter Creme de Violette', 'Rothman & Winter Orchard Apricot', 'Rothman & Winter Orchard Pear', 'Ruby Red Grapefruit Juice', 'Sagatiba Cachaca', 'Sage', 'Sage Leaf', 'Salt', 'San Pellegrino Limonata', 'Sazerac 6-Year-Old Rye Whiskey', 'Schonauer Apfel Schnapps', 'Seedlip', 'Sencha Green Tea-Infused Leblon Cachaca', 'Shinn Estate Rose', 'Shiso Leaves', 'Siembra Azul Blanco Tequila', 'Siembra Azul Reposado Tequila', 'Siete Leguas Blanco Tequila', 'Siete Leguas Reposado Tequila', 'Simple Syrup', 'Small Hand Foods Grapefruit Cordial', 'Smirnoff Black Vodka', 'Smith & Cross Jamaican Rum', 'Sombra Mezcal', 'Southampton Double White Ale', 'Southampton Pumpkin Ale', 'Spiced Macchu Pisco', 'Spiced Sorrel', 'Ssal-Yut Rice Syrup', 'St. Dalfour Fig Jam', 'St. Elizabeth Allspice Dram', 'St. George Absinthe', 'St. Germain Elderflower Liqueur', 'Star Anise Pod', 'Strawberries', 'Strawberry Fan', 'Strawberry Slice', 'Strawberry-Infused Mae de Ouro Cachaca', 'Strega', 'Sugar', 'Sugar Cube', 'Suze', 'Sweetened Whipped Cream', 'Talisker 10-Year-Old Single Malt Scotch Whisky', 'Tamarind Puree', 'Tangerine Zest', 'Tanqueray Gin', 'The Bitter Truth Celery Bitters', 'The Bitter Truth Grapefruit Bitters', 'The Bitter Truth Jerry Thomas Bitters', 'The Bitter Truth Lemon Bitters', 'Theurlet Creme de Cassis', 'Thyme', 'Ting Grapefruit Soda', 'Tokaji Aszu 5 Puttonyos Red Label', 'Tonic Syrup', 'Tonic Water', 'Toro Albala Pedro Ximenez Sherry', 'Trader Tikis Dons Mix', 'Trimbach Framboise', 'Umbrella', 'Vanilla Butter', 'Victory Pilsner', 'Vieux Pontarlier Absinthe', 'Vya Dry Vermouth', 'Vya Sweet Vermouth', 'Walnut-Infused Hine V.S.O.P. Cognac', 'Watermelon Ball', 'Watermelon Juice', 'Whiskey-soaked Goji Berry', 'Whole Black Peppercorn', 'Whole Egg', 'Whole Milk', 'Wild Turkey Russells Reserve 6-Year-Old Rye Whiskey', 'Wild Turkey Rye Whiskey', 'Wilfreds Aperitif', 'Wray & Nephew Overproof Rum', 'Yamazaki 12-Year-Old Japanese Single Malt Whisky', 'Yellow Chartreuse', 'Yogurt', 'Zwack', 'van Oosten Batavia Arrack', 'Mango']
ingredients = sorted(ingredients)

ingredients_nonalcoholic = ['Agave', 'Amass', 'Apple', 'Banana', 'Blackberries', 'Blueberries', 'Buttermilk', 'Club Soda', 'Cocktail Umbrellas', 'Coke', 'Dehydrated Citrus', 'Edible Flowers', 'Grapefruit Juice', 'Honey Syrup', 'Lassi', 'Lavender', 'Lemon', 'Lemon Balm', 'Lemon Juice', 'Lemon and Lime Zest', 'Lime Juice', 'Lyre American Malt', 'Mango Lassi', 'Mape Syrup', 'Maraschino Cherry', 'Mint Leaves', 'Orange', 'Orange Juice', 'Peach', 'Pear', 'Pepsi', 'Pineapple', 'Pineapple Juice', 'Raspberries', 'Ritual Gin', 'Ritual Tequila', 'Rosemary', 'Sage', 'Salt Lassi', 'Seedlip', 'Strawberries', 'Thyme', 'Tonic Water', 'Wilfreds Aperitif', 'Yogurt']
ingredients_nonalcoholic = sorted(ingredients_nonalcoholic)

inspiration = ['#3 Cup', '#8', '100 Year Punch', '20th Century', '212', '21st Century', 'Absinthe Drip', 'Against All Odds Cocktail', 'Aguila Azteca', 'Airmail', 'Albert Mathieu', 'Algonquin', 'Americano Highball', 'Aperol Spritz', 'Apple Daiquiri', 'Apple Malt Toddy', 'Applejack Rabbit', 'Apricot Flip', 'Archangel', 'Astoria Bianco', 'Aviation', 'Beachbum', 'Bees Knees', 'Bees Sip', 'Beer Cassis', 'Beer and a Smoke', 'Bentons Old-Fashioned', 'Berlioni', 'Betsy Ross', 'Betula', 'Bijou', 'Bizet', 'Black Flip', 'Black Jack', 'Black Thorn (Irish)', 'Blackbeard', 'Blackstar', 'Blackthorn (English)', 'Blackthorn Rose', 'Blinker', 'Blood and Sand', 'Bobby Burns', 'Brandy Crusta', 'Brazilian Tea Punch', 'Brewers Breakfast', 'Bronx', 'Brooklyn', 'Brown Bomber', 'Brown Derby', 'Bubbaloo', 'Buona Notte', 'Caipirinha', 'Camerons Kick', 'Caprice', 'Cavalier', 'Champagne Cocktail', 'Champs-Elysees', 'Cherry Pop', 'Chien Chaud', 'Chrysanthemum', 'Cinema Highball', 'Cloister', 'Clover Club', 'Coconut Colada', 'Coda', 'Coke', 'Coffee Cocktail', 'Condiment Cocktail', 'Conquistador', 'Corpse Reviver No. 2', 'Cosmopolitan', 'Cranberry Cobbler', 'Crimson Tide', 'Cuzco', 'Daiquiri', 'De La Louisiane', 'Death Bed', 'Desert Rose', 'Deshler', 'Dewey D.', 'Diamondback', 'Donizetti', 'Dry County Cocktail', 'Duboudreau Cocktail', 'Dulce de Leche', 'East India Cocktail', 'East Village Athletic Club Cocktail', 'Eclipse Cocktail', 'Edgewood', 'El Burro', 'El Diablo', 'El Molino', 'El Puente', 'Ephemeral', 'Espresso Bongo', 'Falling Leaves', 'Field Cocktail', 'Figetaboutit', 'Fish House Punch', 'Flora Astoria', 'Flying Dutchman', 'Fog Cutter', 'Foreign Legion', 'Framboise Fizz', 'Frankfort Rose', 'French 75', 'French Maid', 'Fresa Verde', 'Frisco', 'Gilchrist', 'Gimlet', 'Gin & Tonic', 'Girl from Jerez', 'Gold Coast', 'Gold Rush', 'Golden Star Fizz', 'Great Pumpkin', 'Green Deacon', 'Green Harvest', 'Greenpoint', 'Hanky Panky', 'Harvest Moon', 'Harvest Sling', 'Heirloom', 'Hemingway Daiquiri', 'Henry Hudson', 'Honeymoon Cocktail', 'Hot Buttered Pisco', 'Hotel D Alsace', 'Hotel Nacional Special', 'Imperial Blueberry Fizz', 'Imperial Silver Corn Fizz', 'Improved Whiskey Cocktail', 'Jack Rose', 'Japanese Cocktail', 'Japanese Courage', 'Jimmie Roosevelt', 'Johnny Apple Collins', 'Judgment Day', 'Junior', 'Kansai Kick', 'Kin Kan', 'Kina Miele', 'King Bee', 'Koyo', 'L.E.S. Globetrotter', 'La Florida Cocktail', 'La Louche', 'La Perla', 'Lacrimosa', 'Lake George', 'Last Word', 'Lawn Dart', 'Le Pere Bis', 'Leapfrog', 'Left Coast', 'Left Hand Cocktail', 'Lions Tooth', 'Little Bit Country', 'Luau', 'Mae West Royal Diamond Fizz', 'Mai-Tai', 'Manhattan', 'Margarita', 'Mariner', 'Martinez', 'Martini', 'Mary Pickford', 'Masataka Swizzle', 'Master Cleanse', 'May Daisy', 'May Day', 'Melon Stand', 'Mexicano', 'Mezcal Mule', 'Midnight Express', 'Milk Punch', 'Mint Apple Crisp', 'Mint Julep', 'Mojito', 'Monkey Gland', 'Montgomery Smith', 'Morango Fizz', 'Moscow Mule', 'Mount Vernon', 'Mums Apple Pie', 'Navy Grog', 'Negroni', 'New Amsterdam', 'New York Flip', 'Newark', 'Newfangled', 'Nigori Milk Punch', 'Noce Royale', 'Norman Inversion', 'Nouveau Carre', 'Nouveau Sangaree', 'Noval Cup', 'Nth Degree', 'Occidental', 'Old Flame', 'Old Maid', 'Old Pal', 'Old-Fashioned Whiskey Cocktail', 'Opera Cocktail', 'Paddington', 'Paddy Wallbanger', 'Paloma', 'Parkside Fizz', 'Pauls Club Cocktail', 'Pearl Button', 'Pearl of Puebla', 'Perfect Pear', 'Persephone', 'Pharaoh Cooler', 'Pimms Cup', 'Pink Lady', 'Pisco Sour', 'Platanos en Mole Old Fashioned', 'Primavera', 'Prince Edward', 'Prince of Wales', 'Professor', 'Pumpkin Toddy', 'Queen Park Swizzle', 'Rack & Rye', 'Ramos Gin Fizz', 'Rapscallion', 'Raspberries Reaching', 'Rattlesnake', 'Red Devil', 'Red-headed Saint', 'Remember Maine', 'Remember the Maine', 'Resting Point', 'Reverend Palmer', 'Rhubarbarita', 'Rhum Club', 'Rio Bravo', 'Rite of Spring', 'Rob Roy', 'Romeo Y Julieta', 'Rose', 'Rosita', 'Royal Bermuda Yachtclub Cocktail', 'Rust Belt', 'Rusty Nail', 'Rye Witch', 'Sage Old Buck', 'Sazerac', 'Seelbach Cocktail', 'Shaddock Rose', 'Shiso Delicious', 'Shiso Malt Sour', 'Sidecar', 'Siesta', 'Silk Road', 'Silver Lining', 'Silver Root Beer Fizz', 'Silver Sangaree', 'Singapore Sling', 'Single Malt Sangaree', 'Sloe Gin Fizz', 'Smoky Grove', 'Solstice', 'South Slope', 'Southside', 'Spice Market', 'St. Rita', 'Staggerac', 'Statesman', 'Swiss Mist', 'Swollen Gland', 'T&T', 'Talbott Leaf', 'Tao of Pooh', 'There Will Be Blood', 'Ti-Punch', 'Tipperary Cocktail', 'Tom Collins', 'Tommys Margarita', 'Triborough', 'Trident', 'Tuxedo', 'Up to Date', 'Vaccari', 'Vauvert Slim', 'Velvet Club', 'Vesper', 'Vieux Carre', 'Vieux Mot', 'Ward Eight', 'Water Lily', 'Weeski', 'Wellington Fizz', 'Whiskey Smash', 'White Birch Fizz', 'White Lady', 'White Negroni', 'Widows Kiss', 'Witchs Kiss', 'Woolworth', 'Wrong Aisle', 'Zombie Punch']

inspiration_nonalcoholic = ['Arnold Palmer', 'Cinderella', 'Club Soda & Lime', 'Coconut Water', 'Faux Tropical Fizz', 'Frozen Blackberry Smoothie', 'Ginger Beer', 'Gunner', 'Iced Tea', 'Kombucha', 'Lassi', 'Lemon, Lime & Bitters', 'Lemonade', 'Mango Lassi', 'Mock Champagne', 'Mocktail Mulled Wine', 'No Tequila Sunrise', 'Nojito', 'Non-Alcoholic Irish Creme Liqueur', 'Pineapple & Ginger Punch', 'Roy Rogers', 'Safe Sex On The Beach', 'Shirley Temple', 'Sidecar Mocktail', 'Summer Cup Mocktail', 'Tortuga', 'Virgin Margarita', 'Virgin Mary (Bloody Mary Mocktail)', 'Virgin Moscow Mule', 'Virgin Piña Colada', 'Virgin Strawberry Daiquiri', ]

cuisine_list = ['All Occasions', 'Chinese', 'Greek', 'Indian', 'Italian', 'Japanese', 'American', 'Mexican', 'Thai', 'Mediterranean']
cuisine_list = sorted(cuisine_list)

NON_ALCOHOLIC_FLAG = False
drink = ''

def get_ingredient():
    if(drink == 'Non-Alcoholic'):
        input_text = ', '.join(random.choices(ingredients_nonalcoholic, k=5)) + ", " + ', '.join(optional_ingredient)
    else:
        input_text = ', '.join(random.choices(ingredients, k=5)) + ", " + ', '.join(optional_ingredient)
    return input_text

def get_inspiration(drink):
    if(drink == 'Non-Alcoholic'):
        input_text = random.choices(inspiration_nonalcoholic, k=1)
    else:
        input_text = random.choices(inspiration, k=1)

    input_text = ''.join(input_text)
    return input_text

optional_ingredient = ""

#You can check .empty documentation
placeholder = st.empty()

with placeholder.container():
    with st.form('app'):
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        with col1:
            drink = st.selectbox('Type of drink', options=['Cocktail', 'Shot', 'Punch', 'Non-Alcoholic'])
            print(drink)
            if(drink == 'Non-Alcoholic'):
                NON_ALCOHOLIC_FLAG = True
                drink = 'No Alcohol Mocktail'

        with col2:
            if NON_ALCOHOLIC_FLAG:
                optional_ingredient = st.multiselect(label='I will pick the ingredients. But any particular ones?', options=ingredients_nonalcoholic,)
            else:
                optional_ingredient = st.multiselect(label='I will pick the ingredients. But any particular ones?', options=ingredients,)
            print(optional_ingredient)

        with col4:
            craziness = st.select_slider('How crazy you want me to go?', options=['crazy', 'crazier', 'craziest'])

            if craziness == 'crazier':
                PRESENCE_PENALTY = 1.5
                FREQUENCY_PENALTY = 1.5
            
            if craziness == 'craziest':
                PRESENCE_PENALTY = 2.0
                FREQUENCY_PENALTY = 2.0

        with col3:
            cuisine = st.selectbox('Cusine to pair with', options=cuisine_list)
            print(cuisine)

            if craziness == 'crazier':
                PRESENCE_PENALTY = 1.5
                FREQUENCY_PENALTY = 1.5
            
            if craziness == 'craziest':
                PRESENCE_PENALTY = 2.0
                FREQUENCY_PENALTY = 2.0

        #btn = st.button(label="GENERATE")
        print(NON_ALCOHOLIC_FLAG)
        btn = st.form_submit_button("GENERATE")

    if btn:
        ingredient_input = get_ingredient()
        inspiration_input = get_inspiration(drink)
        
        with st.spinner(text="Building your " + craziness + " " + drink + " recipe ..." + " that pairs well with " + cuisine + " cuisine"):
            output = overall_chain({'drink': drink, 'ingredient': ingredient_input, 'inspiration': inspiration_input, 'cocktail_name': cocktail_name, 'cuisine': cuisine})
            print(output)
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
                ingredients_list = output['cocktail'][:output['cocktail'].index("Rationale")]
                st.markdown(ingredients_list)

            with col2:
                st.subheader("How will this drink look?")
                #st.markdown(drink)
                prompt_4_diffusion = drink + " drink named " + cocktail_name + ". Contains " + ingredient_input + ". Magazine cover --ar 4:3 --v 4 --c 100"
                #st.markdown(prompt_4_diffusion.strip())
                print(prompt_4_diffusion)

                kwargs = {
                    "prompt": prompt_4_diffusion,
                    "n": 1,
                    "size": '512x512'}
                image_resp = openai.Image.create(**kwargs)
                image_url = image_resp['data'][0]['url']
                st.image(image_url)
                st.caption(output['poem'].strip())

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Why this " + drink + "?")
                
                st.multiselect(
                    label='I used the following ingredients in this ' + drink,
                    options=output['ingredient'].split(", "),
                    default=output['ingredient'].split(", "),
                    disabled=True,
                    )

                st.multiselect(
                    label='Inspired from',
                    options=[output['inspiration'].replace(" ", " ")],
                    default=[output['inspiration'].replace(" ", " ")],
                    disabled=True,
                )
                st.markdown(output['cocktail'].strip().partition("Rationale:")[2])
            
            with col2:
                st.subheader("Multi-Chain JSON")
                st.json(output)

#btn_share = st.button("SHARE RECIPE")
#if(btn_share):
#    ss_region = (300, 300, 600, 600)
#    ss_img = ImageGrab.grab(ss_region)
#    ss_img.save("drink.jpg")

st.caption("Non-Humanoid Developer: Swami Chandrasekaran")
