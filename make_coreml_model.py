# Convert CoreML model
import coremltools
import numpy
from keras.models import model_from_json
from keras.utils import np_utils

# load json and create model
json_file = open('weights/model_squeezeNet_TSR.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("weights/model_squeezeNet_TSR.hdf5")
print("Loaded model from disk")

coreml_model = coremltools.converters.keras.convert(model)
coreml_model.save("coreml_model/model_squeezeNet_TSR.mlmodel")
