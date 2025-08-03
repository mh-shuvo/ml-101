import os
import google.generativeai as genai
from dotenv import load_dotenv
# Load environment variables from the .env file
load_dotenv()
def call_gemini():
    """
    Initializes the Gemini model, sends a prompt, and prints the response.
    """
    try:
        # --- 1. Configure the API key ---
        # It's best practice to store your API key in an environment variable.
        # This line retrieves the key from the environment.
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set. Please set it to your API key.")
            
        genai.configure(api_key=api_key)
        
        # list of models
        models = genai.list_models()
        print("Available models:")
        for model in models:
            print(f"- {model.name}")   

        # --- 2. Initialize the Generative Model ---
        # We use GenerativeModel to get an instance of the model.
        print("Initializing the Gemini model...")
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")

        # --- 3. Define your prompt ---
        # This is the input you want to send to the model.
        prompt = "Act as a Python expert. Write a function that takes a list of numbers and returns the sum."
        print(f"Sending prompt: \"{prompt}\"")

        # --- 4. Generate Content ---
        # The generate_content method sends your prompt to the API and waits for the response.
        print("\nGenerating response...")
        response = model.generate_content(prompt)

        # --- 5. Print the Response ---
        # The response object contains the generated text.
        # We add some formatting to make the output clear in the console.
        print("\n--- Gemini's Response ---")
        print(response.text)
        print("-------------------------\n")

    except Exception as e:
        # More specific error handling to catch and display issues.
        print(f"An error occurred: {e}")

# --- Main execution block ---
# This ensures the code inside only runs when the script is executed directly.
# It must be at the top level (no indentation).
if __name__ == "__main__":
    call_gemini()