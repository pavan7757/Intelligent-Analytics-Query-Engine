from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(r'C:\Users\patel\OneDrive\Desktop\Pavan-2315001574/.env'))
import os
from google import genai
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
for model in ['gemini-3.8-flash', 'gemini-3.5-flash', 'gemini-3.5-flash-lite']:
    try:
        response = client.models.generate_content(
            model=model,
            contents='Return only JSON: {"ok": true}',
            config={'response_mime_type': 'application/json'}
        )
        print('MODEL', model, 'OK', type(response).__name__)
        print('ATTRS', [a for a in dir(response) if not a.startswith('_')][:40])
        print('HAS_TEXT', hasattr(response, 'text'))
        print('HAS_PARSED', hasattr(response, 'parsed'))
        print('TEXT', getattr(response, 'text', None))
        print('PARSED', getattr(response, 'parsed', None))
    except Exception as e:
        print('MODEL', model, 'ERR', type(e).__name__, getattr(e, 'status', None), getattr(e, 'code', None), str(e)[:260])
