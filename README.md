# OpenAI Cocktail Recipes Generator 

This is an example generative AI app that will help you create the perfect drink. The user can select the type of drink (cocktail, punch, shot, or non-alcoholic) and optionally provide specific ingredients and the degree of (olfactory) imagination. The app orchestrates a plan including the right set of models (#OpenAI text-davinci-003, text-curie-001 or DALL-E or other) for each of the following tasks including the prompts and context,
1. Generate a drink along with the ingredients (selected by the planner)
2. Instructions to mix the drink
3. Visual image of the drink
4. Restaurant-style caption for the image, and
5. Most importantly, an explanation for why the drink was generated incl why the ingredients were chosen

## Setup

1. If you don’t have Python installed, [install it from here](https://www.python.org/downloads/)

2. Clone this repository

3. Navigate into the project directory

   ```bash
   $ cd cocktails
   ```

4. Create a new virtual environment

   ```bash
   $ python -m venv venv
   $ . venv/bin/activate
   ```

5. Install the requirements

   ```bash
   $ pip install -r requirements.txt
   ```

6. Make a copy of the example environment variables file

   ```bash
   $ cp .env.example .env
   ```

7. Add your [OPENAI API key](https://platform.openai.com/account/api-keys) to the newly created `.env` file

8. Run the app

   ```bash
   $ streamlit run main.py
   ```

You should now be able to access the app at [http://localhost:8501](http://localhost:8501)!

[![Run on Google Cloud](https://deploy.cloud.run/button.svg)](https://deploy.cloud.run)
