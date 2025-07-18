#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

try:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    print("OpenAI client created successfully")
except Exception as e:
    print(f"Error: {e}")
    print(f"Error type: {type(e)}")
    import traceback
    traceback.print_exc()