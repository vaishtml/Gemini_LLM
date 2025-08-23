# LLM_Playground/test_inference.py

import os
import google.generativeai as genai
import argparse

def test_gemini_inference(prompt, model_name):
    """
    Tests the inference of a specific Gemini model using an
    environment variable for authentication.
    """
    # 1. Get API key from environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return

    # 2. Configure the Gemini library
    genai.configure(api_key=api_key)

    print(f"--- Testing model: {model_name} ---")
    try:
        # 3. Create a model instance
        model = genai.GenerativeModel(model_name)

        # 4. Make the API call
        response = model.generate_content(prompt)

        # 5. Print the results
        print("\n[Prompt]")
        print(prompt)
        print("\n[Model Response]")
        print(response.text)
        print("\n--- Test complete ---")

    except Exception as e:
        print(f"An error occurred during the inference test: {e}")

if __name__ == "__main__":
    # Set up to read arguments from the command line
    parser = argparse.ArgumentParser(description="Test Google Gemini Model Inference.")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt to send to the model.")
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-1.5-flash",
        help="The model name to test (e.g., gemini-1.5-flash)."
    )
    
    args = parser.parse_args()
    test_gemini_inference(args.prompt, args.model)