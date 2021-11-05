import logging
import os
import shutil
import sys
import tempfile
import warnings
import time

warnings.simplefilter("ignore")

if sys.platform == 'darwin':
    os.environ['OMP_NUM_THREADS'] = '1'

import matplotlib
import pandas as pd

matplotlib.use('agg')  # no need for tk

from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.core.utils.savers import save_pd, save_pkl
import autogluon.core.metrics as metrics
from autogluon.tabular.version import __version__

from frameworks.shared.callee import call_run, result, output_subdir
from frameworks.shared.utils import Timer, zip_path

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info(f"\n**** AutoGluon [v{__version__}] ****\n")

    metrics_mapping = dict(
        acc=metrics.accuracy,
        auc=metrics.roc_auc,
        f1=metrics.f1,
        logloss=metrics.log_loss,
        mae=metrics.mean_absolute_error,
        mse=metrics.mean_squared_error,
        r2=metrics.r2,
        rmse=metrics.root_mean_squared_error,
    )

    perf_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if perf_metric is None:
        # TODO: figure out if we are going to blindly pass metrics through, or if we use a strict mapping
        log.warning("Performance metric %s not supported.", config.metric)

    is_classification = config.type == 'classification'
    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}
    test_frac = config.framework_params.get('_test_frac', None)
    unlabeled_frac = config.framework_params.get('_unlabeled_frac', None)
    is_pseudo = config.framework_params.get('_use_pseudo', False)
    num_iter = config.framework_params.get('_num_iter', 1)
    is_transductive = config.framework_params.get('_is_transductive', False)
    time_split = 1 if num_iter == 1 else num_iter + 1

    if is_transductive and not is_pseudo:
        raise Exception('\'is_transductive\' is True, but \'is_pseudo\' is False,'
                        ' pseudolabeling must be enabled in order to use transductive learning')

    train, test = dataset.train.path, dataset.test.path
    label = dataset.target.name
    problem_type = dataset.problem_type

    models_dir = tempfile.mkdtemp() + os.sep  # passed to AG

    train_df = TabularDataset(train)
    test_df = TabularDataset(test)

    len_test_df = len(test_df)
    len_all = len_test_df + len(train_df)
    log.info(f"Total data size is {len_all}")

    if test_frac is not None:
        num_test = int(test_frac * len_all)

        if num_test < len_test_df:
            raise Exception('AMLB expects all test data indexes to be returned for scoring'
                            f',but number of requested test rows: {num_test} is less than test length: {len_test_df}')

        log.info(f"Using {test_frac} percent of all data as test or {num_test} rows")

        num_rows_take_from_train = num_test - len_test_df
        add_to_test_df = train_df.sample(num_rows_take_from_train, random_state=0)

        train_df = train_df.drop(add_to_test_df.index)
        test_df = test_df.append(add_to_test_df, verify_integrity=True)

    if unlabeled_frac is not None:
        if is_transductive:
            raise Exception(
                '\'unlabeled_frac\' should only be set when doing semi-supervised, but \'transductive\' is set to true.')
        log.info(f"Using {unlabeled_frac} percent of train data as unlabeled data for pseudolabeling")
        sample_sz_unlabeled = int(unlabeled_frac * len(train_df))
        unlabeled_df = train_df.sample(sample_sz_unlabeled, random_state=0)
        train_df = train_df.drop(unlabeled_df.index)
    else:
        log.info('All test data is used for pseudolabeling fit')
        unlabeled_df = test_df.copy()
    unlabeled_df = unlabeled_df.drop(columns=[label])

    log.info(f"Using {len(train_df)} rows for train")
    log.info(f"Using {len(unlabeled_df)} rows for pseudolabeling")
    log.info(f"Using {len(test_df)} rows for test")

    truth_series = test_df[label]

    log.info(training_params)

    with Timer() as training:
        predictor = TabularPredictor(
            label=label,
            eval_metric=perf_metric.name,
            path=models_dir,
            problem_type=problem_type,
        ).fit(
            train_data=train_df,
            time_limit=config.max_runtime_seconds / time_split,
            **training_params
        )

    if is_pseudo:
        log.info(f"Running Pseudolabel fit with {num_iter} iterations")
        if is_transductive:
            with Timer() as predict:
                predictor, probabilities = predictor.fit_pseudolabel(pseudo_data=unlabeled_df,
                                                                     max_iter=num_iter,
                                                                     return_pred_prob=True,
                                                                     time_limit=config.max_runtime_seconds / time_split,
                                                                     **training_params)
        else:
            predictor = predictor.fit_pseudolabel(test_data=unlabeled_df,
                                                  max_iter=num_iter,
                                                  return_pred_prob=False,
                                                  time_limit=config.max_runtime_seconds / time_split,
                                                  **training_params)
        training.stop = time.time()
    else:
        log.info('No Pseudolabeling used')
        probabilities = None

    del train

    if is_classification:
        if not is_transductive:
            with Timer() as predict:
                probabilities = predictor.predict_proba(test_df, as_multiclass=True)
        predictions = probabilities.idxmax(axis=1).to_numpy()
    else:
        if is_transductive:
            predictions = probabilities
        else:
            with Timer() as predict:
                predictions = predictor.predict(test_df, as_pandas=False)
        probabilities = None

    prob_labels = probabilities.columns.values.astype(str).tolist() if probabilities is not None else None

    _leaderboard_extra_info = config.framework_params.get('_leaderboard_extra_info',
                                                          False)  # whether to get extra model info (very verbose)
    _leaderboard_test = config.framework_params.get('_leaderboard_test',
                                                    False)  # whether to compute test scores in leaderboard (expensive)
    leaderboard_kwargs = dict(silent=True, extra_info=_leaderboard_extra_info)
    # Disabled leaderboard test data input by default to avoid long running computation, remove 7200s timeout limitation to re-enable
    if _leaderboard_test:
        leaderboard_kwargs['data'] = test_df

    leaderboard = predictor.leaderboard(**leaderboard_kwargs)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        log.info(leaderboard)

    num_models_trained = len(leaderboard)
    if predictor._trainer.model_best is not None:
        num_models_ensemble = len(predictor._trainer.get_minimum_model_set(predictor._trainer.model_best))
    else:
        num_models_ensemble = 1

    save_artifacts(predictor, leaderboard, config)
    shutil.rmtree(predictor.path, ignore_errors=True)

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  probabilities=probabilities,
                  probabilities_labels=prob_labels,
                  target_is_encoded=False,
                  models_count=num_models_trained,
                  models_ensemble_count=num_models_ensemble,
                  training_duration=training.duration,
                  predict_duration=predict.duration,
                  truth=truth_series)


def save_artifacts(predictor, leaderboard, config):
    artifacts = config.framework_params.get('_save_artifacts', ['leaderboard'])
    try:
        if 'leaderboard' in artifacts:
            leaderboard_dir = output_subdir("leaderboard", config)
            save_pd.save(path=os.path.join(leaderboard_dir, "leaderboard.csv"), df=leaderboard)

        if 'info' in artifacts:
            ag_info = predictor.info()
            info_dir = output_subdir("info", config)
            save_pkl.save(path=os.path.join(info_dir, "info.pkl"), object=ag_info)

        if 'models' in artifacts:
            shutil.rmtree(os.path.join(predictor.path, "utils"), ignore_errors=True)
            models_dir = output_subdir("models", config)
            zip_path(predictor.path, os.path.join(models_dir, "models.zip"))
    except Exception:
        log.warning("Error when saving artifacts.", exc_info=True)


if __name__ == '__main__':
    call_run(run)
