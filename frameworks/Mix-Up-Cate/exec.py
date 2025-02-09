import logging
import os
import shutil
import sys
import tempfile
import warnings

from imblearn.over_sampling import BorderlineSMOTE

warnings.simplefilter("ignore")

if sys.platform == 'darwin':
    os.environ['OMP_NUM_THREADS'] = '1'

import matplotlib
import pandas as pd
import numpy as np

matplotlib.use('agg')  # no need for tk

from autogluon.tabular import TabularPredictor, TabularDataset
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
    is_data_aug = config.framework_params.get('_is_data_aug', False)

    train, test = dataset.train.path, dataset.test.path
    label = dataset.target.name
    problem_type = dataset.problem_type

    if is_data_aug:
        train_data = TabularDataset(train)
        test_data = TabularDataset(test)

        predictor_og = TabularPredictor(label=label).fit(train_data, time_limit=10)
        X, y, X_val, y_val, X_unlabeled, holdout_frac, num_bag_folds, groups = predictor_og._learner.general_data_processing(
            train_data, test_data, test_data, 0, 1)

        train_data = X.copy()
        train_data[label] = y

        if predictor_og.problem_type == 'regression':
            # for feat in categorical_features:
            #     train_data[feat] = pd.to_numeric(train_data[feat])
            #
            # for feat in categorical_features:
            #     X_unlabeled[feat] = pd.to_numeric(X_unlabeled[feat])
            #
            # num_samples = int(len(train_data) / 2)
            # train_sample_1 = train_data.sample(num_samples).reset_index(drop=True)
            # train_sample_2 = train_data.sample(num_samples).reset_index(drop=True)
            # lam = np.random.beta(0.4, 0.4, num_samples)[:, None].repeat(len(train_data.columns), axis=1)
            #
            # train_data_mixed = lam * train_sample_1 + (1 - lam) * train_sample_2
            #
            # train_data.append(train_data_mixed).reset_index(drop=True)
            # else:
            # num_samples = int(len(train_data) / 4)
            # train_sample_1 = train_data.sample(num_samples).reset_index(drop=True)
            # train_sample_2 = train_data.sample(num_samples).reset_index(drop=True)
            # lam = np.ones(train_sample_1.shape) * 0.5
            #
            # train_data_mixed = lam * train_sample_1 + (1 - lam) * train_sample_2
            # train_data_mixed[label] = train_sample_1[label]
            # train_data.append(train_data_mixed)
            # train_data_mixed[label] = train_sample_2[label]
            # train_data.append(train_data_mixed).reset_index(drop=True)
            numerical_features = train_data.columns[train_data.dtypes != 'category']
            categorical_features = train_data.columns[train_data.dtypes == 'category']
            #
            if not categorical_features.empty:
                grouped_df = train_data.groupby(by=list(categorical_features))
                mixed_rows_df = None
                for key, value in grouped_df.groups.items():
                    num_rows = len(value)
                    if num_rows < 2:
                        continue

                    if num_rows % 2 != 0:
                        num_rows -= 1

                    selected_rows = train_data.loc[value[:num_rows]]

                    half_num_rows = int(num_rows / 2)

                    sample_1_df = selected_rows.iloc[:half_num_rows].reset_index(drop=True)
                    sample_2_df = selected_rows.iloc[half_num_rows:].reset_index(drop=True)

                    lam = np.random.beta(0.4, 0.4, half_num_rows)[:, None].repeat(len(numerical_features), axis=1)

                    new_mixed_rows_df = lam * sample_1_df[numerical_features] + (1 - lam) * sample_2_df[
                        numerical_features]

                    if mixed_rows_df is not None:
                        mixed_rows_df = mixed_rows_df.append(new_mixed_rows_df, ignore_index=True)
                    else:
                        mixed_rows_df = new_mixed_rows_df

                log.info(f'Adding in {len(mixed_rows_df)} rows of mix-up')
                train_data = train_data.append(mixed_rows_df, ignore_index=True).reset_index(drop=True)
        else:
            X_resampled, y_resampled = BorderlineSMOTE().fit_resample(X, y)
            resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
            resampled_df[label] = pd.Series(y_resampled)

            log.info(f'Adding in {len(resampled_df)} rows of SMOTE')
            train_data = train_data.append(resampled_df, ignore_index=True).reset_index(drop=True)

        test = X_val.copy()
        train = train_data
        y_truth = y_val
    else:
        y_truth = test[label]

    models_dir = tempfile.mkdtemp() + os.sep  # passed to AG

    with Timer() as training:
        predictor = TabularPredictor(
            label=label,
            eval_metric=perf_metric.name,
            path=models_dir,
            problem_type=problem_type,
        ).fit(
            train_data=train,
            time_limit=config.max_runtime_seconds,
            **training_params
        )

    del train

    if is_classification:
        with Timer() as predict:
            probabilities = predictor.predict_proba(test, as_multiclass=True)
        predictions = probabilities.idxmax(axis=1).to_numpy()
    else:
        with Timer() as predict:
            predictions = predictor.predict(test, as_pandas=False)
        probabilities = None

    prob_labels = probabilities.columns.values.astype(str).tolist() if probabilities is not None else None

    _leaderboard_extra_info = config.framework_params.get('_leaderboard_extra_info',
                                                          False)  # whether to get extra model info (very verbose)
    _leaderboard_test = config.framework_params.get('_leaderboard_test',
                                                    False)  # whether to compute test scores in leaderboard (expensive)
    leaderboard_kwargs = dict(silent=True, extra_info=_leaderboard_extra_info)
    # Disabled leaderboard test data input by default to avoid long running computation, remove 7200s timeout limitation to re-enable
    if _leaderboard_test:
        leaderboard_kwargs['data'] = test

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
                  truth=y_truth)


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
