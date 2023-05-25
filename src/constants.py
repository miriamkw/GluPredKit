from dotenv import load_dotenv
import inspect
import models
from models.base_model import BaseModel

# This is some convenience logic for automatically exposing the names of
# all the implemented models that inherit from BaseModel. Currently I'm
# just using this to validate --model-name arguments coming into the cli
# ...feel free to delete or replace.
VALID_MODELS = []
for name, obj in inspect.getmembers(models):
    if inspect.isclass(obj) and issubclass(obj, BaseModel):
        VALID_MODELS.append(name)

# Your .env.local file should include the secret parameters needed to
# access your tidepool and/or nightscout account... something like this:
# NIGHTSCOUT_URL=<my_nighscout_url>
# NIGHTSCOUT_API_SECRET=<my_nightscout_secret>
# TIDEPOOL_USERNAME=<my_tidepool_un>
# TIDEPOOL_PASSWORD=<my_tidepool_pw>
# the following code exposes those environment variables to this program:
load_dotenv()
load_dotenv(dotenv_path=".env.local", override=True)


