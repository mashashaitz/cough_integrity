from copy import deepcopy
import random

from tabulate import tabulate


def pretty_print_df(df, column_number=4, rnd_state=10, reject_col=False):
    # pretty print dataset
    columns = list(df.columns)

    # pick 10 random entries
    short_df = deepcopy(df).sample(n=10, random_state=rnd_state).sort_index()

    # for embeddings pick 4 random features to display
    if 'feature_1' in columns:
        random.seed(rnd_state + 1)
        random_columns = sorted(random.sample(columns, column_number))

        if 'filename' in columns:
            if 'filename' in random_columns:
                random_columns.remove('filename')
                random_columns.append('filename')
            else:
                random_columns[-1] = 'filename'

        short_df = short_df[random_columns]

    # for meta pick 4 important columns to display
    elif 'is_completed' in columns:
        if not reject_col:
            important_columns = ['duration', 'is_completed', 'reject_id', 'filename']
        else:
            important_columns = ['is_completed', 'reject_id', 'filename', 'rejected']
        short_df = short_df[important_columns]

    return tabulate(short_df, headers='keys', tablefmt='psql')


def pretty_print_all_clf_options(clf_final_estimator):
    # print all of the versions of classifiers
    means = clf_final_estimator.cv_results_['mean_test_score']
    stds = clf_final_estimator.cv_results_['std_test_score']
    results = []
    for mean, std, params in zip(means, stds,
                                 clf_final_estimator.cv_results_['params']):
        results.append("%0.3f (+/-%0.03f) for %r\n" % (mean, std * 2, params))

    return '\n'.join(results)
