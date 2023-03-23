import pickle
import tempfile
import time
import zipfile
from pathlib import Path

import mlflow


class func_to_class(mlflow.pyfunc.PythonModel):
    def __init__(self, func):
        self.func = func

    def predict(self, context, model_input):
        return self.func(model_input)


track = 2
dataset = 1


with open(f'submission_data/pickles/model_{track}_{dataset}.pickle', 'rb') as handle:
    compact_model = pickle.load(handle)

assert compact_model


def predict(seq):
    current_state = 's0'
    for i in seq[1:-1]:
        current_state = compact_model[current_state][1][i]
    return compact_model[current_state][0]


#zip_path = f"submission_data/model_{track}_{dataset}_{int(time.time())}.zip"
zip_path = f"submission_data/model_{track}_{dataset}.zip"
with tempfile.TemporaryDirectory() as tmp_dir:
    mlflow_path = Path(tmp_dir)

    mlflow.pyfunc.save_model(
        path=mlflow_path,
        python_model=func_to_class(predict),
    )

    with zipfile.ZipFile(zip_path, "w") as zip_file:
        for f in mlflow_path.rglob('*'):
            if f.is_file():
                zip_file.write(f, f.relative_to(mlflow_path))

print(f'Submission file: {zip_path}')
