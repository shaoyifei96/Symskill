from google import genai
from google.genai import types
import enum
import os

class Instrument(enum.Enum):
  BLOCK = "Block"
  PLATE = "Plate"
  LID = "Lid"
  DISHRACK = "Dishrack"
  PAN = "Pan"
  BANANA = "Banana"

client = genai.Client(api_key=os.getenv('GEMINI_KEY'))

# Load an image file
with open('/home/yifei/Documents/task_planning_2/feature_data/vlm_trajectories/vlm_motion_0_12_lid_type_MocapMulti.png', 'rb') as image_file:
    image_data1 = image_file.read()
# Load an image file
with open('/home/yifei/Documents/task_planning_2/feature_data/vlm_trajectories/vlm_motion_0_8_lid_type_MocapMulti.png', 'rb') as image_file:
    image_data2 = image_file.read()

response = client.models.generate_content(
    model='gemini-2.5-pro',
    contents=[
        {
            'parts': [
                {'text': 'Which item is the object moving to in the image?'},
                types.Part.from_bytes(data = image_data1, mime_type='image/png'),
                types.Part.from_bytes(data = image_data2, mime_type='image/png'),
            ]
        }
    ],
    config={
        'response_mime_type': 'text/x.enum',
        'response_schema': Instrument,
    },
)

print(response.text)
# Woodwind