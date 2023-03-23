import os
import profile
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import mlflow
from memory_profiler import profile


class func_to_class(mlflow.pyfunc.PythonModel):
    def __init__(self, func):
        self.func = func

    def predict(self, context, model_input):
        return self.func(model_input)


model_name = f"models/{1}.{5}.taysir.model"

model = mlflow.pytorch.load_model(model_name)
model.eval()


def predict(seq):
    do_something = 1
    do_something_2 = [i for i in range(100)]
    return model.predict(model.one_hot_encode(seq))
    # return False


zip_path = f"submission_data/model_test.zip"
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

# unzip in tmp folder
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('tmp')

# load the model
loaded_model = mlflow.pyfunc.load_model(model_uri='tmp/')
unwrapped_model = loaded_model.unwrap_python_model()

# remove tmp subfolder
shutil.rmtree('tmp/')

# execute few sequences to check if loading worked
print('Model loaded, testing it on 10 sequences')

# here you load validation data input sequances
validation_data = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]

log_file = open('log.txt', 'w+')


# define an outer function, as it logs are written to file only after its context is over
def outer_function():
    # Mib Precision to 6 decimal palaces, as small models could be 0
    @profile(precision=6, stream=log_file)
    # some function that executes tests, and it includes a call to predict function
    # this one can also return total time in ms
    def execute_tests():
        cum_time = 0

        for seq in validation_data[:10]:
            t = time.time_ns() // 1_000_000
            # Context is None?
            o = unwrapped_model.predict(None, seq)
            # check against reference
            cum_time += (time.time_ns() // 1_000_000) - t

        return cum_time

    cum_time = execute_tests()

    return cum_time


cum_time = outer_function()

# this you need to set to a line which is calling predict, so you get a memory usage of this line only
predict_function = 'o = unwrapped_model.predict(None, seq)'

memory_usage = None

with open('log.txt') as log_file:
    for line in log_file:
        if predict_function in line:
            memory_usage = float(line.split('MiB')[1].split('MiB')[0])

assert memory_usage
print(f'Total time in predict function   : {cum_time}ms')
print(f'Memory usage in predict function : {memory_usage}MiB')

# os.remove('log.txt')
