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

# All of Streamlit config and customization
st.set_page_config(page_title="Cocktail Maker powered by Generative AI", page_icon=":random:", layout="wide")
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
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

#START LLM portions
if os.environ["OPENAI_API_KEY"]:
    st.title("Cocktail Maker")
    st.caption("Let generative AI come up with drink recipes")
else:
    st.error("🔑 Please enter API Key")

cocktail_name = ""

# Available Models
LANGUAGE_MODELS = ['gpt-3.5-turbo', 'text-davinci-003', 'text-curie-001', 'text-babbage-001', 'text-ada-001']
CODEX_MODELS = ['code-davinci-002', 'code-cushman-001']
PRIMARY_MODEL = 'gpt-3.5-turbo' #"text-davinci-002"
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

template = """I want someone who can suggest out of the world and imaginative drink recipes. You are my master mixologist. You will come up with crazy and bold alcoholic {drink} that is appealing, attractive, imaginative and never seen before. Use {ingredient} in your recipe. Draw inspiration from an existing cocktail recipe of {inspiration}. Give the drink a unique name. Ingredients must start in a new line. Add a catch phrase for the drink within double quotes. Provide a scientific explanation for why the ingredients were chosen. 
Cocktail Name: 
Ingredients:
Instructions:
Rationale:###
"""

prompt_4_cocktail = PromptTemplate(input_variables=["drink", "ingredient", "inspiration"],template=template.strip(),)
cocktail_gen_chain = LLMChain(llm=llm, prompt=prompt_4_cocktail, output_key="cocktail", verbose=True)

#This is an LLMChain to generate a short haiku poem for the cocktail based on the ingredients.
llm = OpenAI(model_name=SECONDARY_MODEL, temperature=0.7, frequency_penalty=FREQ_PENALTY, presence_penalty=PRESENCE_PENALTY, max_tokens=200, best_of=3, top_p=0.5)

template2 = """###Write a restaurant menu style short description for a {drink} that has the following ingredients {ingredient}###."""

prompt_4_poem = PromptTemplate(input_variables=["drink", "ingredient"], template=template2.strip(),)
cocktail_poem_chain = LLMChain(llm=llm, prompt=prompt_4_poem, output_key="poem", verbose=True)

#This is the overall chain where we run these two chains in sequence.
overall_chain = SequentialChain(
    chains=[cocktail_gen_chain, cocktail_poem_chain],
    input_variables=['drink', 'ingredient', 'inspiration'],
    # Here we return multiple variables
    output_variables=['cocktail', 'poem'],
    verbose=True)
#END LLM portions


# From here down is all the StreamLit UI.
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

ingredients = ['7-Up', 'Absinthe', 'Absolut Citron', 'Absolut Kurant', 'Absolut Peppar', 'Absolut Vodka', 'Advocaat', 'Agave Syrup', 'Agave syrup', 'Ale', 'Allspice', 'Almond', 'Almond flavoring', 'Amaretto', 'Angelica root', 'Angostura Bitters', 'Anis', 'Anise', 'Anisette', 'Aperol', 'Apfelkorn', 'Apple', 'Apple brandy', 'Apple cider', 'Apple juice', 'Apple schnapps', 'Applejack', 'Apricot', 'Apricot brandy', 'Aquavit', 'Asafoetida', 'Bacardi Limon', 'Baileys irish cream', 'Banana', 'Banana liqueur', 'Beer', 'Benedictine', 'Berries', 'Bitter lemon', 'Bitters', 'Black pepper', 'Black Sambuca', 'Blackberry brandy', 'Blackcurrant cordial', 'Blackcurrant squash', 'Blended whiskey', 'Blue Curacao', 'Blueberry schnapps', 'Bourbon', 'Brandy', 'Brown sugar', 'Butter', 'Butterscotch schnapps', 'Cachaca', 'Campari', 'Candy', 'Cantaloupe', 'Caramel coloring', 'caramel sauce', 'Carbonated soft drink', 'Carbonated water', 'Cardamom', 'Carrot', 'Cayenne pepper', 'Celery salt', 'Chambord raspberry liqueur', 'Champagne', 'Cherry', 'Cherry brandy', 'Cherry Grenadine', 'Cherry Heering', 'Cherry liqueur', 'Chocolate', 'Chocolate ice-cream', 'Chocolate liqueur', 'Chocolate milk', 'chocolate sauce', 'Chocolate syrup', 'Cider', 'Cinnamon', 'Cloves', 'Club soda', 'Coca-Cola', 'Cocoa powder', 'Coconut liqueur', 'Coconut milk', 'Coconut rum', 'Coconut syrup', 'Coffee', 'Coffee brandy', 'Coffee liqueur', 'Cognac', 'Cointreau', 'Condensed milk', 'Coriander', 'Corn syrup', 'Cornstarch', 'Corona', 'Cranberries', 'Cranberry juice', 'Cranberry vodka', 'Cream', 'Cream of coconut', 'Creme de Banane', 'Creme de Cacao', 'Creme de Cassis', 'Creme de Mure', 'Crown Royal', 'Cumin seed', 'Curacao', 'Daiquiri mix', 'Dark Creme de Cacao', 'Dark rum', 'Dark Rum', 'demerara Sugar', 'Dr. Pepper', 'Drambuie', 'Dry Vermouth', 'Dubonnet Rouge', 'Egg', 'Egg white', 'Egg yolk', 'Erin Cream', 'Espresso', 'Everclear', 'Fennel seeds', 'Firewater', 'Food coloring', 'Frangelico', 'Fresca', 'Fresh Lemon Juice', 'Fresh Lime Juice', 'Fruit juice', 'Fruit punch', 'Galliano', 'Gin', 'Ginger', 'Ginger ale', 'Ginger beer', 'Ginger Beer', 'Glycerine', 'Godiva liqueur', 'Gold tequila', 'Goldschlager', 'Grain alcohol', 'Grand Marnier', 'Grape juice', 'Grape soda', 'Grapefruit juice', 'Grapes', 'Green Chartreuse', 'Green Creme de Menthe', 'Grenadine', 'Guava juice', 'Guinness stout', 'Half-and-half', 'Heavy cream', 'Honey', 'Hot chocolate', 'Hot Damn', 'Ice', 'Iced tea', 'Irish cream', 'Irish whiskey', 'Jack Daniels', 'Jello', 'Jim Beam', 'Johnnie Walker', 'Kahlua', 'Kirschwasser', 'Kiwi', 'Kiwi liqueur', 'Kool-Aid', 'Kummel', 'Lager', 'Lavender', 'lemon', 'Lemon juice', 'Lemon peel', 'Lemon vodka', 'Lemon-lime soda', 'Lemonade', 'Licorice root', 'Light cream', 'Light rum', 'Lillet Blanc', 'Lime', 'Lime juice', 'Lime juice cordial', 'Lime peel', 'Lime vodka', 'Limeade', 'Malibu rum', 'Mango', 'Maple syrup', 'Maraschino cherry', 'Maraschino Liqueur', 'Marjoram leaves', 'Marshmallows', 'Maui', 'Melon liqueur', 'Midori melon liqueur', 'Milk', 'Mini-snickers bars', 'Mint', 'Mint syrup', 'Mountain Dew', 'Nutmeg', 'Olive', 'Olive Brine', 'Orange', 'Orange bitters', 'Orange Curacao', 'Orange juice', 'Orange peel', 'Orange spiral', 'Oreo cookie', 'Orgeat syrup', 'Ouzo', 'Papaya', 'Passion fruit juice', 'Passion fruit syrup', 'Peach Bitters', 'Peach brandy', 'Peach nectar', 'Peach schnapps', 'Peach Vodka', 'Peachtree schnapps', 'Pepper', 'Peppermint extract', 'Peppermint schnapps', 'Pepsi Cola', 'Peychaud bitters', 'Pina colada mix', 'Pineapple', 'Pineapple juice', 'Pineapple Syrup', 'Pink lemonade', 'Pisang Ambon', 'Pisco', 'Port', 'Powdered sugar', 'Prosecco', 'Raspberry Liqueur', 'Raspberry syrup', 'Raspberry vodka', 'Red wine', 'Ricard', 'Root beer', 'Rum', 'Rum', 'Rumple Minze', 'Rye whiskey', 'Salt', 'Sambuca', 'Sarsaparilla', 'Schweppes Russchian', 'Scotch', 'Sherry', 'Sirup of roses', 'Sloe gin', 'Soda water', 'Soda Water', 'Sour mix', 'Southern Comfort', 'Spiced rum', 'Sprite', 'St. Germain', 'Strawberries', 'Strawberry liqueur', 'Strawberry schnapps', 'Sugar', 'Sugar syrup', 'Surge', 'Sweet and sour', 'Sweet Vermouth', 'Tabasco sauce', 'Tea', 'Tennessee whiskey', 'Tequila', 'Tia maria', 'Tomato juice', 'Tomato Juice', 'Tonic water', 'Triple Sec', 'Tropicana', 'Vanilla extract', 'Vanilla Ice-Cream', 'Vanilla Vodka', 'Vermouth', 'Vodka', 'Water', 'Whipped cream', 'Whiskey', 'White Creme de Menthe', 'White Rum', 'Wild Turkey', 'Wine', 'Worcestershire sauce', 'Wormwood', 'Yellow Chartreuse', 'Yogurt', 'Yukon Jack', 'Zima']
inspiration = ['1-900-FUK-MEUP', '110 in the shade', '151 Florida Bushwacker', '155 Belmont', '24k nightmare', '252', '3 Wise Men', '3-Mile Long Island Iced Tea', '410 Gone', '50/50', '501 Blue', '57 Chevy with a White License Plate', '69 Special', '747', '9 1/2 Weeks', 'A Day at the Beach', 'A Furlong Too Late', 'A Gilligans Island', 'A Night In Old Mandalay', 'A Piece of Ass', 'A Splash of Nash', 'A True Amaretto Sour', 'A.D.M. (After Dinner Mint)', 'A1', 'ABC', 'ACID', 'Abbey Cocktail', 'Abbey Martini', 'Abilene', 'Absinthe No.2', 'Absolut Evergreen', 'Absolut Sex', 'Absolut Stress No.2', 'Absolut Summertime', 'Absolut limousine', 'Absolutely Cranberry Smash', 'Absolutely Fabulous', 'Absolutly Screwed Up', 'Acapulco', 'Ace', 'Adam', 'Adam Bomb', 'Adam Sunrise', 'Adam and Eve', 'Addington', 'Addison', 'Addison Special', 'Adios Amigos Cocktail', 'Adonis Cocktail', 'Affair', 'Affinity', 'After Dinner Cocktail', 'After Five', 'After Supper Cocktail', 'After sex', 'Afterglow', 'Afternoon', 'Alabama Slammer', 'Alaska Cocktail', 'Alexander', 'Alfie Cocktail', 'Algonquin', 'Alice Cocktail', 'Alice in Wonderland', 'Allegheny', 'Allies Cocktail', 'Almeria', 'Almond Chocolate Coffee', 'Almond Joy', 'Aloha Fruit punch', 'Amaretto And Cream', 'Amaretto Liqueur', 'Amaretto Mist', 'Amaretto Rose', 'Amaretto Shake', 'Amaretto Sour', 'Amaretto Stinger', 'Amaretto Stone Sour', 'Amaretto Stone Sour No.3', 'Amaretto Sunrise', 'Amaretto Sunset', 'Amaretto Sweet and Sour', 'Amaretto Tea', 'Americano', 'Angel Face', 'Angelica Liqueur', 'Apello', 'Apple Berry Smoothie', 'Apple Cider Punch No.1', 'Apple Grande', 'Apple Karate', 'Apple Pie with A Crust', 'Apple Slammer', 'Applecar', 'Applejack', 'Apricot Lady', 'Apricot punch', 'Archbishop', 'Arctic Fish', 'Arctic Mouthwash', 'Arise My Love', 'Arizona Antifreeze', 'Arizona Stingers', 'Arizona Twister', 'Army special', 'Arthur Tompkins', 'Artillery', 'Artillery Punch', 'Atlantic Sun', 'Atomic Lokade', 'Auburn Headbanger', 'Avalanche', 'Avalon', 'Aviation', 'Aztec Punch', 'B-52', 'B-53', 'Baby Eskimo', 'Baby Guinness', 'Bacardi Cocktail', 'Baileys Dream Shake', 'Balmoral', 'Banana Cantaloupe Smoothie', 'Banana Daiquiri', 'Banana Milk Shake', 'Banana Strawberry Shake', 'Banana Strawberry Shake Daiquiri-type', 'Barracuda', 'Belgian Blue', 'Bellini', 'Bellini Martini', 'Bermuda Highball', 'Berry Deadly', 'Between The Sheets', 'Bible Belt', 'Big Red', 'Black Forest Shake', 'Black Russian', 'Black and Brown', 'Black and Tan', 'Blackthorn', 'Bleeding Surgeon', 'Blind Russian', 'Bloody Maria', 'Bloody Mary', 'Blue Lagoon', 'Blue Margarita', 'Blue Mountain', 'Bluebird', 'Bob Marley', 'Bobby Burns Cocktail', 'Boomerang', 'Boozy Snickers Milkshake', 'Bora Bora', 'Boston Sidecar', 'Boston Sour', 'Bourbon Sling', 'Bourbon Sour', 'Boxcar', 'Brain Fart', 'Brainteaser', 'Bramble', 'Brandon and Wills Coke Float', 'Brandy Alexander', 'Brandy Cobbler', 'Brandy Flip', 'Brandy Sour', 'Brave Bull Shooter', 'Bruces Puce', 'Bruised Heart', 'Bubble Gum', 'Buccaneer', 'Bumble Bee No.1', 'Butter Baby', 'Cafe Savoy', 'Caipirinha', 'Caipirissima', 'California Lemonade', 'California Root Beer', 'Campari Beer', 'Caribbean Boilermaker', 'Caribbean Orange Liqueur', 'Casino', 'Casino Royale', 'Castillian Hot Chocolate', 'Champagne Cocktail', 'Cherry Electric Lemonade', 'Cherry Rum', 'Chicago Fizz', 'Chocolate Beverage', 'Chocolate Black Russian', 'Chocolate Drink', 'Chocolate Milk', 'Chocolate Monkey', 'Citrus Coke', 'City Slicker', 'Classic Old-Fashioned', 'Clove Cocktail', 'Clover Club', 'Coffee Liqueur', 'Coffee-Vodka', 'Coke and Drops', 'Cosmopolitan', 'Cosmopolitan Martini', 'Cranberry Cordial', 'Cranberry Punch', 'Cream Soda', 'Creme de Menthe', 'Cuba Libra', 'Cuba Libre', 'Daiquiri', 'Damned if you do', 'Danbooka', 'Dark Caipirinha', 'Dark and Stormy', 'Darkwood Sling', 'Derby', 'Diesel', 'Dirty Martini', 'Dirty Nipple', 'Downshift', 'Dragonfly', 'Drinking Chocolate', 'Dry Rob Roy', 'Dubonnet Cocktail', 'Duchamps Punch', 'Egg Cream', 'Egg Nog - Healthy', 'Egg Nog No.4', 'Egg-Nog - Classic Cooked', 'English Highball', 'English Rose Cocktail', 'Espresso Martini', 'Fahrenheit 5000', 'Flaming Dr. Pepper', 'Flaming Lamborghini', 'Flanders Flake-Out', 'Flying Dutchman', 'Flying Scotchman', 'Foxy Lady', 'Freddy Kruger', 'French "75"', 'French 75', 'French Connection', 'French Martini', 'Frisco Sour', 'Frozen Daiquiri', 'Frozen Mint Daiquiri', 'Frozen Pineapple Daiquiri', 'Fruit Cooler', 'Fruit Flip-Flop', 'Fruit Shake', 'Fuzzy Asshole', 'GG', 'Gagliardo', 'Gentlemans Club', 'Gideons Green Dinosaur', 'Gin And Tonic', 'Gin Cooler', 'Gin Daisy', 'Gin Fizz', 'Gin Rickey', 'Gin Sling', 'Gin Smash', 'Gin Sour', 'Gin Squirt', 'Gin Swizzle', 'Gin Toddy', 'Girl From Ipanema', 'Gluehwein', 'Godchild', 'Godfather', 'Godmother', 'Golden dream', 'Grand Blue', 'Grape lemon pineapple Smoothie', 'Grass Skirt', 'Grasshopper', 'Green Goblin', 'Grim Reaper', 'Grizzly Bear', 'H.D.', 'Happy Skipper', 'Harvey Wallbanger', 'Havana Cocktail', 'Hawaiian Cocktail', 'Hemingway Special', 'Herbal flame', 'Highland Fling Cocktail', 'Holloween Punch', 'Homemade Kahlua', 'Horses Neck', 'Hot Chocolate to Die for', 'Hot Creamy Bush', 'Ice Pick No.1', 'Iced Coffee', 'Iced Coffee Fillip', 'Imperial Cocktail', 'Imperial Fizz', 'Ipamena', 'Irish Coffee', 'Irish Cream', 'Irish Curdling Cow', 'Irish Russian', 'Irish Spring', 'Jack Rose Cocktail', 'Jackhammer', 'Jacks Vanilla Coke', 'Jam Donut', 'Jamaica Kiss', 'Jamaican Coffee', 'Japanese Fizz', 'Jello shots', 'Jelly Bean', 'Jewel Of The Nile', 'Jitterbug', 'John Collins', 'Just a Moonmint', 'Kamikaze', 'Karsk', 'Kentucky B And B', 'Kentucky Colonel', 'Kill the cold Smoothie', 'Kioki Coffee', 'Kir', 'Kir Royale', 'Kiss me Quick', 'Kiwi Lemon', 'Kiwi Papaya Smoothie', 'Kool First Aid', 'Kool-Aid Shot', 'Kool-Aid Slammer', 'Kurant Tea', 'Lady Love Fizz', 'Lassi - A South Indian Drink', 'Lassi - Mango', 'Lassi - Sweet', 'Lassi Khara', 'Lassi Raita', 'Lemon Drop', 'Lemon Shot', 'Lemouroudji', 'Limeade', 'Limona Corona', 'Loch Lomond', 'London Town', 'Lone Tree Cocktail', 'Lone Tree Cooler', 'Long Island Iced Tea', 'Long Island Tea', 'Long vodka', 'Lord And Lady', 'Lunch Box', 'Mai Tai', 'Malibu Twister', 'Mango Orange Smoothie', 'Manhattan', 'Margarita', 'Martinez Cocktail', 'Martini', 'Mary Pickford', 'Masala Chai', 'Melya', 'Miami Vice', 'Microwave Hot Cocoa', 'Midnight Cowboy', 'Midnight Manx', 'Midnight Mint', 'Mimosa', 'Mississippi Planters Punch', 'Mocha-Berry', 'Mojito', 'Mojito No.3', 'Monkey Gland', 'Monkey Wrench', 'Moranguito', 'Moscow Mule', 'Mothers Milk', 'Mudslinger', 'Mulled Wine', 'National Aquarium', 'Negroni', 'New York Lemonade', 'New York Sour', 'Nuked Hot Chocolate', 'Nutty Irishman', 'Old Fashioned', 'Orange Crush', 'Orange Oasis', 'Orange Push-up', 'Orange Scented Hot Chocolate', 'Orange Whip', 'Orangeade', 'Oreo Mudslide', 'Orgasm', 'Owens Grandmothers Revenge', 'Paradise', 'Pina Colada', 'Pineapple Gingerale Smoothie', 'Pink Gin', 'Pink Lady', 'Pink Panty Pulldowns', 'Pink Penocha', 'Pisco Sour', 'Planters Punch', 'Popped cherry', 'Poppy Cocktail', 'Port And Starboard', 'Port Wine Cocktail', 'Port Wine Flip', 'Porto flip', 'Pysch Vitamin Light', 'Quakers Cocktail', 'Quarter Deck Cocktail', 'Queen Bee', 'Queen Charlotte', 'Queen Elizabeth', 'Quentin', 'Quick F**K', 'Quick-sand', 'Radioactive Long Island Iced Tea', 'Radler', 'Rail Splitter', 'Raspberry Cooler', 'Red Snapper', 'Rose', 'Royal Bitch', 'Royal Fizz', 'Royal Flush', 'Royal Gin Fizz', 'Ruby Tuesday', 'Rum Cobbler', 'Rum Cooler', 'Rum Milk Punch', 'Rum Old-fashioned', 'Rum Punch', 'Rum Runner', 'Rum Screwdriver', 'Rum Sour', 'Rum Toddy', 'Russian Spring Punch', 'Rusty Nail', 'Salty Dog', 'San Francisco ', 'Sangria No.1', 'Sazerac', 'Scooter', 'Scotch Cobbler ', 'Scotch Sour', 'Scottish Highland Liqueur ', 'Screaming Orgasm', 'Screwdriver', 'Sea Breeze', 'Sex on the Beach ', 'Shanghai Cocktail', 'Shark Attack ', 'Sherry Eggnog', 'Sherry Flip ', 'Shot-gun', 'Sidecar', 'Sidecar Cocktail', 'Singapore Sling ', 'Sloe Gin Cocktail', 'Smut ', 'Snake Bite (UK)', 'Snakebite and Black', 'Snowball', 'Sol Y Sombra ', 'Space Odyssey', 'Spanish Chocolate ', 'Spiced Peach Punch', 'Spiking Coffee ', 'Spritz', 'Stinger', 'Stone Sour', 'Strawberry Daiquiri ', 'Strawberry Lemonade', 'Strawberry Margarita ', 'Strawberry Shivers', 'Sunny Holiday Punch ', 'Surf City Lifesaver', 'Swedish Coffee ', 'Sweet Bananas', 'Sweet Sangria ', 'Sweet Tooth', 'Talos Coffee', 'Tennesee Mud', 'Tequila Fizz', 'Tequila Sour', 'Tequila Sunrise', 'Tequila Surprise', 'Texas Rattlesnake', 'Texas Sling', 'Thai Coffee', 'Thai Iced Coffee', 'Thai Iced Tea', 'The Evil Blue Thing', 'Thriller', 'Tia-Maria', 'Tom Collins', 'Tomato Tang', 'Tommys Margarita', 'Turf Cocktail', 'Turkeyball', 'Tuxedo Cocktail', 'Valencia Cocktail', 'Vampiro', 'Van Vleet', 'Vermouth Cassis', 'Vesper', 'Vesuvio', 'Veteran', 'Victor', 'Victory Collins', 'Vodka And Tonic', 'Vodka Fizz', 'Vodka Martini', 'Vodka Russian', 'Waikiki Beachcomber', 'Whiskey Sour', 'Whisky Mac', 'White Lady', 'White Russian', 'Whitecap Margarita', 'Wine Cooler', 'Wine Punch', 'Yellow Bird', 'Yoghurt Cooler', 'Zambeer', 'Zenmeister', 'Ziemes Martini Apfelsaft', 'Zima Blaster', 'Zimadori Zinger', 'Zinger', 'Zipperhead', 'Zippys Revenge', 'Zizi Coin-coin', 'Zoksel', 'Zorbatini', 'Zorro']

def get_ingredient():
    input_text = ', '.join(random.choices(ingredients, k=5)) + ", " + ', '.join(optional_ingredient)
    return input_text

def get_inspiration():
    input_text = random.choices(inspiration, k=1)
    input_text = ''.join(input_text)
    return input_text

optional_ingredient = ""
with st.form('app'):
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        #optional_ingredient = st.text_input('Any particular ingredients?', placeholder="Add as many ingredients you want me to use seperated by commas")
        optional_ingredient = st.multiselect(label='Any ingredients you want me to include?', options=ingredients,)
        print(optional_ingredient)

    with col2:
        craziness = st.select_slider('How crazy you want me to go?', options=['crazy', 'crazier', 'craziest'])

        if craziness == 'crazier':
            PRESENCE_PENALTY = 1.5
            FREQUENCY_PENALTY = 1.5
        
        if craziness == 'craziest':
            PRESENCE_PENALTY = 2.0
            FREQUENCY_PENALTY = 2.0

    with col3:
        drink = st.selectbox('Type of drink', options=['Cocktail', 'Shot', 'Punch'])
        print(drink)
  
    #btn = st.button(label="GENERATE")
    btn = st.form_submit_button("GENERATE")

if btn:
    ingredient_input = get_ingredient()
    inspiration_input = get_inspiration()
    
    with st.spinner(text="Building your " + craziness + " " + drink + " recipe ..."):
        output = overall_chain({'drink': drink, 'ingredient': ingredient_input, 'inspiration': inspiration_input, 'cocktail_name': cocktail_name })
        print(output)
        cocktail_name = output['cocktail'][:output['cocktail'].index("Ingredients")]
        cocktail_name = cocktail_name.strip().partition("Cocktail Name:")[2].strip()
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
            prompt_4_diffusion = "Backlight alcoholic " + drink + " drink named " + cocktail_name + ". Layered, earthy colors -uplight --s 42000 --ar 4:3 --version 3"
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
            st.multiselect(
                label='Ingredients used in this ' + drink,
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

            st.subheader("Why this " + drink + "?")
            st.markdown(output['cocktail'].strip().partition("Rationale:")[2])
        
        with col2:
            st.subheader("Multi-Chain JSON")
            st.json(output)

st.caption("Non-Humanoid Developer: Swami Chandrasekaran")
