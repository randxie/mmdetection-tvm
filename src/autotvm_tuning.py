import os

import numpy as np

import tvm
import tvm.contrib.graph_runtime as runtime
import tvm.relay.testing
from constants import DEPLOY_WEIGHT_DIR
from constants import TUNING_LOG_DIR
from tvm import autotvm
from tvm import relay
from tvm.autotvm.tuner import GATuner
from tvm.autotvm.tuner import GridSearchTuner
from tvm.autotvm.tuner import RandomTuner
from tvm.autotvm.tuner import XGBTuner
from tvm.contrib.util import tempdir


def get_network(batch_size=1):
  # load exported parameters, graph def and library.
  input_shape = (batch_size, 3, 300, 300)
  output_shape = (batch_size, 8732, 6)

  export_ssd_module = os.path.join(DEPLOY_WEIGHT_DIR, "ssd_module.json")
  export_params = os.path.join(DEPLOY_WEIGHT_DIR, "ssd_param.params")

  module = tvm.ir.load_json(open(export_ssd_module, "r").read())
  params = relay.load_param_dict(open(export_params, "rb").read())

  return module, params, input_shape, output_shape


#### DEVICE CONFIG ####
target = 'llvm'

# Set number of threads used for tuning based on the number of
# physical CPU cores on your machine.
num_threads = 1
os.environ["TVM_NUM_THREADS"] = str(num_threads)

#### TUNING OPTION ####
log_file = os.path.join(TUNING_LOG_DIR, "ssd_tuning.log")
dtype = 'float32'

tuning_option = {
  'log_filename': log_file,
  'tuner': 'xgb',
  'n_trial': 2000,
  'early_stopping': 600,
  'measure_option': autotvm.measure_option(
    builder=autotvm.LocalBuilder(timeout=10),
    runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
  ),
}


# You can skip the implementation of this function for this tutorial.
def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename=os.path.join(TUNING_LOG_DIR, 'ssd_tuning.log'),
               use_transfer_learning=False):
  # create tmp log file
  tmp_log_file = log_filename + ".tmp"
  if os.path.exists(tmp_log_file):
    os.remove(tmp_log_file)

  for i, tsk in enumerate(reversed(tasks)):
    prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

    # create tuner
    if tuner == 'xgb' or tuner == 'xgb-rank':
      tuner_obj = XGBTuner(tsk, loss_type='rank')
    elif tuner == 'ga':
      tuner_obj = GATuner(tsk, pop_size=100)
    elif tuner == 'random':
      tuner_obj = RandomTuner(tsk)
    elif tuner == 'gridsearch':
      tuner_obj = GridSearchTuner(tsk)
    else:
      raise ValueError("Invalid tuner: " + tuner)

    if use_transfer_learning:
      if os.path.isfile(tmp_log_file):
        tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

    # do tuning
    tsk_trial = min(n_trial, len(tsk.config_space))
    tuner_obj.tune(n_trial=tsk_trial,
                   early_stopping=early_stopping,
                   measure_option=measure_option,
                   callbacks=[
                     autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                     autotvm.callback.log_to_file(tmp_log_file)
                   ])

  # pick best records to a cache file
  autotvm.record.pick_best(tmp_log_file, log_filename)
  os.remove(tmp_log_file)


def tune_and_evaluate(tuning_opt):
  # extract workloads from relay program
  print("Extract tasks...")
  mod, params, input_shape, out_shape = get_network(batch_size=1)

  # set ops to None to extract all tunable parameters
  tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

  # run tuning tasks
  print("Tuning...")
  tune_tasks(tasks, **tuning_opt)

  # compile kernels with history best records
  with autotvm.apply_history_best(log_file):
    print("Compile...")
    with tvm.transform.PassContext(opt_level=3):
      graph, lib, params = relay.build_module.build(
        mod, target=target, params=params)

    # export library
    tmp = tempdir()
    filename = "net.tar"
    lib.export_library(tmp.relpath(filename))

    # load parameters
    ctx = tvm.context(str(target), 0)
    module = runtime.create(graph, lib, ctx)
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    module.set_input('input0', data_tvm)
    module.set_input(**params)

    # evaluate
    print("Evaluate inference time cost...")
    ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=600)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
          (np.mean(prof_res), np.std(prof_res)))


# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.

tune_and_evaluate(tuning_option)
