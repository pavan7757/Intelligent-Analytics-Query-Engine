from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(r'C:\Users\patel\OneDrive\Desktop\Pavan-2315001574/.env'))
import os
from google import genai
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
models = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.5-flash-lite', 'gemini-2.0-flash-lite']
for model in models:
    try:
        response = client.models.generate_content(
            model=model,
            contents='Return only JSON: {"ok": true}',
            config={'response_mime_type': 'application/json'}
        )
        print('MODEL', model, 'OK', type(response).__name__, 'has_text', hasattr(response, 'text'), 'has_candidates', hasattr(response, 'candidates'))
        if hasattr(response, 'text'):
            print('TEXT_PREFIX', str(response.text)[:80])
    except Exception as e:
        print('MODEL', model, 'ERR', type(e).__name__, getattr(e, 'status', None), getattr(e, 'code', None), str(e)[:220])
