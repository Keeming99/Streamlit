import torch
import streamlit as st
from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import difflib

import os
from transformers import LlamaTokenizer

# Retrieve the Hugging Face token from Streamlit secrets
hf_token = st.secrets["huggingface"]["token"]

# Use the token for loading the tokenizer
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_auth_token=hf_token)

# Class for IntentResponseInference (from the original code)
class IntentResponseInference:
    def __init__(self, base_model_path, adapter_path):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = LlamaForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=quantization_config,
            device_map='auto'
        )
        
        self.model = PeftModel.from_pretrained(self.model, adapter_path)

        # Define the list of possible intents (from the original code)
        self.intents = [
    # Account Management
    "create_account", "close_account", "update_account_information", "reset_password", "recover_username",
    "change_password", "login_issues", "account_lockout", "manage_notifications", "privacy_settings",

    # Product Information
    "product_availability", "product_availability_in_store", "product_availability_online", 
    "product_information", "product_comparison", "product_specifications", "product_reviews",
    "product_recommendations", "product_sizes", "product_colors", "product_price", "check_discounts",
    
    # Order Management
    "place_order", "track_order", "cancel_order", "change_order", "order_status",
    "order_history", "repeat_order", "preorder_product", "order_confirmation",
    
    # Payment Issues
    "payment_methods", "payment_failed", "billing_information", "change_payment_method", 
    "redeem_coupon", "gift_card_balance", "invoice_request", "payment_security",

    # Returns and Refunds
    "return_policy", "return_product", "return_in_store", "return_online",
    "request_refund", "refund_policy", "refund_status", "exchange_product",
    
    # Shipping and Delivery
    "shipping_options", "delivery_time", "shipping_costs", "track_delivery", "delivery_status",
    "delivery_issue", "delayed_delivery", "damaged_delivery", "missing_item", "pickup_in_store",
    
    # Account Services
    "customer_service", "speak_to_agent", "contact_support", "store_location", 
    "store_hours", "chat_with_agent", "escalate_issue", "technical_issue", "use_app",

    # Promotions and Discounts
    "apply_promotion", "available_discounts", "promo_code_not_working", "loyalty_program",
    "redeem_rewards", "sales_period",

    # Product Feedback and Complaints
    "submit_feedback", "submit_product_feedback", "report_product_issue", "report_website_issue",
    "complain_about_service", "request_improvement",

    # Notifications and Updates
    "order_update_notification", "shipping_update_notification", "promotion_notification",
    "new_arrival_notification",

    # General Inquiries
    "store_location", "store_opening_hours", "available_products", "check_stock_in_store",
    "gift_wrapping_options", "how_to_use_product", "size_guide", "care_instructions",

    # Assistance Requests
    "human_agent", "need_help", "find_product", "find_suitable_size", "product_recommendation",
    
    # Other Services
    "gift_card_purchase", "gift_card_balance_inquiry", "bulk_order_inquiry", "custom_order_request",
    "subscription_services", "rewards_program_details", "request_catalog",

    # Technical Issues
    "app_crashing", "website_not_loading", "payment_gateway_error", "technical_support", "password_reset_link_not_working",

    # Legal and Privacy
    "request_right_to_rectification", "request_data_deletion", "privacy_policy", "terms_and_conditions",
    "data_usage", "cookie_preferences",

    # Miscellaneous
    "add_product_to_cart", "remove_product_from_cart", "wishlist_management", "compare_products",
    "request_invoice", "request_quote", "manage_saved_addresses", "update_contact_details",
    "track_package", "lost_package", "report_fraud", "request_call_back", "accessibility_support",
    "report_missing_feature", "suggest_feature", "ask_for_demo"
        ]
        
    def generate_response(self, instruction):
        prompt = f"""Instruction: {instruction}
Please provide a brief and concise response.
Response:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            inputs.input_ids, 
            max_length=200,  # Limit response length
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9
        )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Basic parsing
        try:
            parts = generated_text.split('Response:')
            if len(parts) > 1:
                response = parts[1].strip()
                predicted_intent = self.get_closest_intent(instruction)
                
                return {
                    'prompt': instruction,
                    'predicted_intent': predicted_intent,
                    'response': response
                }
        except Exception:
            pass
        
        return {
            'prompt': instruction,
            'predicted_intent': 'Unknown',
            'response': generated_text
        }

    def get_closest_intent(self, instruction):
        # Find the closest matching intent from the list of possible intents
        closest_matches = difflib.get_close_matches(instruction.lower(), self.intents, n=1, cutoff=0.4)
        return closest_matches[0] if closest_matches else 'Unknown'

# New Conversational Chatbot Class
class ConversationalChatbot:
    def __init__(self, base_model_path, adapter_path):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb4bit_compute_dtype=torch.float16,
            bnb4bit_quant_type="nf4",
            bnb4bit_use_double_quant=True
        )
        
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = LlamaForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=quantization_config,
            device_map='auto'
        )
        
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        
        # Initialize session state for conversation history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

    def generate_response(self, user_input):
        # Limit context to last 5 exchanges to prevent context overflow
        context = "\n".join([
            f"{'Human' if i % 2 == 0 else 'AI'}: {msg.strip()}" 
            for i, msg in enumerate(st.session_state.chat_history[-10:])
        ])
        
        prompt = f"""{context}
Human: {user_input}
AI: """
        
        # Ensure input is on the same device as the model
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
        
        # Generate response with more controlled parameters
        outputs = self.model.generate(
            inputs.input_ids, 
            max_length=len(inputs.input_ids[0]) + 100,  # Dynamically adjust max length
            num_return_sequences=1,
            temperature=0.7,  # Slightly reduced for more coherent responses
            top_p=0.9,
            top_k=50,  # Added top_k for additional diversity control
            do_sample=True,  # Enable sampling
            no_repeat_ngram_size=2  # Prevent repetition
        )
        
        # Decode the generated response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the AI's response
        try:
            # Split by the most recent context and take the last part
            response = generated_text.split(f"{user_input}\nAI:")[-1].strip()
            response = response.split("Human:")[0].strip()
        except:
            response = "I apologize, but I couldn't generate a proper response."
        
        return response

    def chat(self):
        st.title("Conversational Chatbot")
    
        # Initialize chat history in session state if not exists
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
    
        # Initialize input tracker in session state
        if 'input_key' not in st.session_state:
            st.session_state.input_key = 0
    
        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            role = "🧑 Human:" if i % 2 == 0 else "🤖 AI:"
            st.write(f"{role} {message}")
    
        # Text input with a dynamic key to force reset
        user_input = st.text_input(
            "Type your message:", 
            key=f"chat_input_{st.session_state.input_key}"
        )
    
        if st.button("Send"):
            if user_input.strip():
                # Add user message to history
                st.session_state.chat_history.append(user_input)
            
            # Generate assistant's response
                try:
                    assistant_response = self.generate_response(user_input)
                    st.session_state.chat_history.append(assistant_response)
                except Exception as e:
                    st.error(f"Error generating response: {e}")
            
            # Increment input key to reset input
                st.session_state.input_key += 1
            else:
                st.warning("Please enter a message!")

# Streamlit UI Tabs
def home_tab():
    st.title("E-commerce Customer Support Chatbot")
    user_input = st.text_input("Enter your question:")
    if st.button("Submit"):
        result = intent_generator.generate_response(user_input)
        st.write("Prompt:", result['prompt'])
        st.write("Predicted Intent:", result['predicted_intent'])
        st.write("Response:", result['response'])

def chatbot_info_tab():
    st.title("Chatbot Information")
    st.write("This chatbot is powered by a Large Language Model trained on e-commerce customer support data.")
    st.write("It can handle a variety of questions and issues related to our e-commerce platform.")

def feedback_tab():
    st.title("Provide Feedback")
    rating = st.slider("How helpful was the chatbot?", 1, 5, 3)
    comments = st.text_area("Additional comments:")
    if st.button("Submit Feedback"):
        st.write("Thank you for your feedback!")

def faqs_tab():
    st.title("Frequently Asked Questions")
    st.write("1. How do I place an order?")
    st.write("Answer: To place an order, go to the product page and click the 'Add to Cart' button. Then, proceed to the checkout page to complete your purchase.")
    # Add more FAQs here

def clear_chat_history():
    st.session_state.chat_history = []

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Home", "Intent Chatbot", "Conversational Chat", "Chatbot Info", "Feedback", "FAQs"])

    # Add a clear chat history button for the conversational chat
    if selection == "Conversational Chat":
        if st.sidebar.button("Clear Chat History"):
            clear_chat_history()

    if selection == "Home":
        home_tab()
    elif selection == "Intent Chatbot":
        home_tab()
    elif selection == "Conversational Chat":
        conversational_chatbot.chat()
    elif selection == "Chatbot Info":
        chatbot_info_tab()
    elif selection == "Feedback":
        feedback_tab()
    elif selection == "FAQs":
        faqs_tab()

if __name__ == "__main__":
    intent_generator = IntentResponseInference(
        base_model_path="meta-llama/Llama-2-7b-chat-hf", 
        adapter_path="./final_model"
    )
    conversational_chatbot = ConversationalChatbot(
        base_model_path="meta-llama/Llama-2-7b-chat-hf", 
        adapter_path="./final_model"
    )
    main()

