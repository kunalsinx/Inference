import numpy as np
import json
from keras.models import model_from_json

with open('net.json') as f:
    json_string = json.load(f)

model = model_from_json(json_string)
model.load_weights('net.h5')
model.save('case.h5')

# X_hit = np.load('hit_shape.npy')
# X_info = np.load('hit_info.npy')
# y = np.load('target.npy')

# y_pred = model.predict([X_hit, X_info])