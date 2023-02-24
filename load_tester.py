

# load the model
import mlflow
import zipfile

from extraction import get_validation_data

zip_file = 'dataset1.0_1677274548.zip'
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall('tmp')

loaded_model = mlflow.pyfunc.load_model(model_uri='tmp/')
print(type(loaded_model)) # <class 'mlflow.pyfunc.model.PyFuncModel'>

unwrapped_model = loaded_model.unwrap_python_model()
validation_data, start_symbol, end_symbol = get_validation_data(1, 0)

for seq in validation_data:
    o = unwrapped_model.predict(None, seq)
    print(o)