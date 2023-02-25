import logging
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from random import choices

import mlflow

logging.getLogger("mlflow").setLevel(logging.DEBUG)


class func_to_class(mlflow.pyfunc.PythonModel):
    def __init__(self, func):
        self.func = func

    def predict(self, context, model_input):
        return self.func(model_input)


def save_function(func, alphabet_size, prefix, start_symbol, end_symbol):
    print('Testing function...')
    alphabet = list(range(0, alphabet_size))
    alphabet.remove(start_symbol)
    alphabet.remove(end_symbol)
    sample_input = choices(alphabet, k=25)
    output = func(sample_input)
    try:
        float(output)
    except ValueError:
        raise ValueError(f'The output should be an integer or a float, but we found a {type(output).__name__}')
    print('Test  passed.')

    print('Creating submission...')
    with tempfile.TemporaryDirectory() as tmp_dir:
        mlflow_path = Path(tmp_dir)
        zip_path = f"{prefix}_{int(time.time())}.zip"

        code_paths = list(Path().rglob('*.py'))

        mlflow.pyfunc.save_model(
            path=mlflow_path,
            python_model=func_to_class(func),
            code_path=code_paths
        )

        with zipfile.ZipFile(zip_path, "w") as zip_file:
            for f in mlflow_path.rglob('*'):
                if f.is_file():
                    zip_file.write(f, f.relative_to(mlflow_path))
    print(f'Submission created at {zip_path}.')
    print('You can now submit your model on the competition website.')


def load_function(path_to_zip, validation_data):

    # unzip in tmp folder
    with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
        zip_ref.extractall('tmp')

    # load the model
    loaded_model = mlflow.pyfunc.load_model(model_uri='tmp/')
    unwrapped_model = loaded_model.unwrap_python_model()

    # execute few sequences to check if loading worked
    print('Model loaded, testing it on 10 sequences')
    for seq in validation_data[:10]:
        # Context is None?
        o = unwrapped_model.predict(None, seq)
        print(o)

    # remove tmp subfolder
    import shutil
    shutil.rmtree('tmp/')
