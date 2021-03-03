import os
import random
from copy import deepcopy
import warnings
from collections import Counter
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import utils
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.exceptions import NotFittedError
from scikitplot.metrics import plot_roc
import joblib

from pretty_prints import pretty_print_df, pretty_print_all_clf_options
from supporting_functions import fname2patient, fname2rec, emb_f2fname, fname2db, rec2rec_type


class CoughClassifier(object):
    def __init__(self, meta_paths, embeddings_paths, log_path, clf_path, auc_path,
                 clf_type='regular', emb_type='average', n_components=20, rnd_state=34, **kwargs):
        self.log_path = log_path
        self.clf_path = clf_path
        self.auc_path = auc_path

        self.clf_type = clf_type
        self.emb_type = emb_type

        logging.basicConfig(filename=self.log_path,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.INFO)

        self.meta_paths = meta_paths
        self.embeddings_paths = embeddings_paths
        self.rnd_state = rnd_state

        if 'parameters' in kwargs:
            self.parameters = kwargs['parameters']
        else:
            self.parameters = {
                'kernel': ('linear', 'rbf', 'sigmoid'),
                'gamma': [1e-3, 1e-4],
                'C': [1, 10, 100, 1000],
                'random_state': [self.rnd_state]}

            logging.info(f'classifier data is:\n'
                         f'{", ".join([p.split("/")[-1].split(".")[0] for p in self.meta_paths])}\n'
                         f'\t\t\t\t\t\t\t\t\tclassifier type is {self.clf_type}\n'
                         f'\t\t\t\t\t\t\t\t\tmultiple embedding dissociation type is {self.emb_type}\n'
                         f'\t\t\t\t\t\t\t\t\tinitial random state is {self.rnd_state}')

        self.meta_df = pd.DataFrame()
        self.embeddings_df = pd.DataFrame()
        self.x = pd.DataFrame()
        self.y = pd.DataFrame()
        self.patients = []
        self.x_train, self.y_train, self.train_patients = pd.DataFrame(), [], []
        self.x_test, self.y_test, self.test_patients = pd.DataFrame(), [], []

        self.n_components = n_components

        if 'clf' in kwargs:
            self.clf = deepcopy(kwargs['clf'])
        else:
            self.clf = make_pipeline(StandardScaler(),
                                     PCA(svd_solver='randomized',
                                         random_state=deepcopy(self.rnd_state)),
                                     GridSearchCV(svm.SVC(class_weight='balanced', probability=True,
                                                          random_state=deepcopy(self.rnd_state)),
                                                  deepcopy(self.parameters),
                                                  cv=self.custom_cv(),
                                                  scoring='roc_auc', n_jobs=-1, verbose=2))

        self.properties = {}

    @staticmethod
    def read_meta_data(meta_path, indexes, used_filenames):

        # get the meta data
        file_formats = ['.m4a', '.aac', '.mp3', '.wav', '.webm', '.ogg']

        meta_df = pd.read_csv(meta_path)
        if 'id' not in meta_df.columns:
            meta_df['id'] = range(len(meta_df))

        new_indexes = []
        for index in meta_df.id:
            if int(index) in indexes or index in new_indexes:
                if not new_indexes and not indexes:
                    new_indexes.append(1)
                elif not new_indexes:
                    new_indexes.append(int(max(indexes) + 1))
                elif not indexes:
                    new_indexes.append(int(max(new_indexes)) + 1)
                else:
                    new_indexes.append(int(max(max(new_indexes), max(indexes))) + 1)
            else:
                new_indexes.append(int(index))

        meta_df.id = new_indexes

        for filename in meta_df.filename:
            if filename not in used_filenames:
                used_filenames.append(filename)
            else:
                logging.info(f'a file named {filename} was used twice')

        clean_filenames = []
        for filename in meta_df.filename:
            clean_filename = filename
            for file_format in file_formats:
                clean_filename = clean_filename.replace(file_format, '')
            clean_filenames.append(clean_filename)
        meta_df.filename = clean_filenames

        # Source5 scenario
        if 'rec_duration' in meta_df.columns:
            meta_df = meta_df[[test is not True for test in list(meta_df.test)]]
            meta_df['duration'] = meta_df.rec_duration

            is_completeds = []
            reject_ids = []
            for row in meta_df.iterrows():
                score = row[1]["labeling"]
                if score >= 3.5:
                    reject_ids.append(0)
                    is_completeds.append(1)
                elif score < 3.5:
                    reject_ids.append(6)
                    is_completeds.append(1)
                else:
                    reject_ids.append(0)
                    is_completeds.append(0)
            meta_df['reject_id'] = reject_ids
            meta_df['is_completed'] = is_completeds

        return meta_df

    def collect_files(self, path):

        # gather the files
        filenames = []
        for root, dirs, files in os.walk(path):
            if 'Source6' in path:
                for d in dirs:
                    for filename in list(self.meta_df.filename):
                        if d in filename:
                            filenames.append(path + filename.replace('_', '/') + '.npy')
            else:
                for filename in files:
                    if 'NoiseCheck' not in filename:
                        if filename.replace('.npy', '') in list(self.meta_df.filename):
                            filenames.append(path + filename)

        return filenames

    @staticmethod
    def files2array(files, emb_type):

        # download the embeddings
        array = np.empty(shape=(0, 512))
        for file in files:
            emb_f = np.load(file, encoding='ASCII')
            if emb_f.shape[0] == 1:
                array = np.concatenate((array, emb_f))
            else:
                if emb_type == 'median':
                    array = np.concatenate((array, np.nanmedian(emb_f, axis=0).reshape(1, -1)))
                if emb_type == 'average':
                    array = np.concatenate((array, np.average(emb_f, axis=0).reshape(1, -1)))
                if emb_type == 'middle':
                    mid_idx = int(emb_f.shape[0] / 2)
                    array = np.concatenate((array, emb_f[mid_idx].reshape(1, -1)))
        return array

    def read_embeddings_data(self, embeddings_path):
        # get the embeddings

        # gather the files
        embedding_files = self.collect_files(embeddings_path)
        embedding_filenames = [emb_f2fname(filename) for filename in embedding_files]

        # download the embeddings
        embeddings = self.files2array(embedding_files, emb_type=self.emb_type)

        # prep the dataframe

        edf_index = []
        for filename in embedding_filenames:
            if filename not in list(self.meta_df.filename):
                edf_index.append(max(self.meta_df.index) + len(edf_index) + 1)
            else:
                edf_index.append(self.meta_df.loc[self.meta_df['filename'] == filename].index[0])

        embeddings_df = pd.DataFrame(embeddings,
                                     index=edf_index,
                                     columns=['feature_' + str(i + 1) for i in range(512)])
        embeddings_df['filename'] = embedding_filenames
        embeddings_df.index.name = 'id'

        return embeddings_df

    def get_all_meta_dfs(self, complex_split=False, printout_df=True):

        # get all dataframes together
        important_columns = ['duration', 'is_completed', 'reject_id', 'filename', 'id']
        self.meta_df = pd.DataFrame(columns=important_columns)

        for meta_path in self.meta_paths:
            self.meta_df = pd.concat([self.meta_df,
                                      self.read_meta_data(meta_path=meta_path, indexes=list(self.meta_df.id),
                                                          used_filenames=list(self.meta_df.filename)
                                                          )[important_columns]], sort=False)
        self.meta_df.set_index('id', drop=True, inplace=True)

        # pretty print datasets
        if printout_df:
            logging.info('meta dataframe:\n'
                         f'{pretty_print_df(self.meta_df, rnd_state=self.rnd_state)}')

        # to make the split more even lets only take counting from Source2 dataset
        if complex_split:
            good_filenames = list(self.meta_df.filename)
            for row in self.meta_df.iterrows():
                if rec2rec_type(fname2rec(row[1].filename)) == 'counting'\
                        and not fname2db(row[1].filename) in ['Source2', 'Source5']:
                    good_filenames.remove(row[1].filename)
            self.meta_df = self.meta_df[self.meta_df.filename.isin(good_filenames)]

    def get_all_embedding_dfs(self, printout_df=True):

        # get all dataframes together
        self.embeddings_df = pd.DataFrame(columns=['feature_' + str(i + 1) for i in range(512)])

        for embeddings_path in self.embeddings_paths:
            self.embeddings_df = pd.concat([self.embeddings_df,
                                            self.read_embeddings_data(embeddings_path)],
                                           sort=False)

        # pretty print datasets
        if printout_df:
            logging.info('embeddings dataframe:\n'
                         f'{pretty_print_df(self.embeddings_df, rnd_state=self.rnd_state)}')

    @staticmethod
    def get_accepted_rejected(meta_df, rnd_state, clf_type):
        # find all rejected files and accepted coughs
        rejected = []
        accepted = []
        patients = []
        recordings = []

        # let's not have more than 1000 accepted Source2 files
        source2_counter = 0

        # if the audio is not corrupted
        for row in utils.shuffle(deepcopy(meta_df), random_state=rnd_state).iterrows():
            if row[1].is_completed == 1 and row[1].duration > 0:
                filename = row[1].filename
                patient = fname2patient(filename)
                recording = fname2rec(filename)

                if clf_type == 'regular':
                    # add all files with suitable reject_id column values to rejected
                    if row[1].reject_id in [1, 2, 4, 5, 6] and recording == 'cough':
                        rejected.append(filename)
                        patients.append(patient)
                        recordings.append(recording)

                    # add all files with Cough in their name to accepted
                    elif recording == 'cough' and row[1].reject_id != 3:
                        if fname2db(filename) != 'Source2':
                            accepted.append(filename)
                            patients.append(patient)
                            recordings.append(recording)
                        elif fname2db(filename) == 'Source2' and source2_counter < 1000:
                            accepted.append(filename)
                            patients.append(patient)
                            recordings.append(recording)
                            source2_counter += 1

                elif clf_type == 'validity':
                    if row[1].reject_id not in [1, 2, 3, 4, 5, 6] and 'ough' in filename:
                        accepted.append(filename)
                        patients.append(patient)
                        recordings.append(recording)

                elif clf_type == 'quality':
                    # add all cough files with suitable reject_id column values to rejected
                    if row[1].reject_id in [1, 2, 4, 5, 6] and 'ough' in filename:
                        rejected.append(filename)
                        patients.append(patient)
                        recordings.append(recording)

                    # add all accepted cough files to accepted
                    elif 'ough' in filename and row[1].reject_id != 3 and len(accepted) < len(rejected):
                        accepted.append(filename)
                        patients.append(patient)
                        recordings.append(recording)

        logging.info(
            f'\n\t\t\t\t\t\t\t\t\taccepted {str(len(accepted))} cough files'
            f'\n\t\t\t\t\t\t\t\t\trejected {str(len(rejected))} cough files'
            f'\n\t\t\t\t\t\t\t\t\tthere are {str(len(np.unique(patients)))} unique patients')

        return accepted, rejected, patients, recordings

    @staticmethod
    def augmentation(meta_df, accepted, rejected, recordings, clf_type, rnd_state=10):
        # add other filetypes as rejected
        phonemes = {'ah1': 0, 'ah2': 0, 'ah3': 0, 'u1': 0, 'u2': 0, 'other_phoneme': 0,
                    'ss1': 0, 'ss2': 0, 'ss3': 0, 'zz1': 0, 'zz2': 0, 'zz3': 0}
        unique_recording_types = {'counting': 0, 'other': 0, 'reading': 0, 'phoneme': 0}
        # unique_patients = np.unique(patients)
        mean_num_recording_types = (len(accepted)-len(rejected)) / 4
        mean_num_phonemes = mean_num_recording_types/len(phonemes.keys())

        augmented_patients = [fname2patient(filename) for filename in rejected]
        augmented_recordings = deepcopy(recordings)
        augmented_rejected = deepcopy(rejected)
        unfit_recordings = []
        accepted_patients = [fname2patient(filename) for filename in accepted]

        accepted_sources = {}
        for filename in accepted:
            source = fname2db(filename)
            if source in accepted_sources:
                accepted_sources[source] += 1
            else:
                accepted_sources[source] = 1
        augmented_sources = {}
        for filename in rejected:
            source = fname2db(filename)
            if source in augmented_sources:
                augmented_sources[source] += 1
            else:
                augmented_sources[source] = 1

        permutation = 0
        # if the audio is not corrupted
        while len(augmented_rejected) < len(accepted) and \
                len(augmented_rejected) + len(accepted) + len(unfit_recordings) < len(meta_df):
            if len(augmented_rejected) != len(rejected):
                logging.info(
                    f'rejected {str(len(augmented_rejected))} files in augmentation()')
            permutation += 1
            if permutation > 15:
                logging.info('Unexpected break\n'
                             f'rejected {str(len(augmented_rejected))} files in augmentation()\n'
                             f'file types: {str(Counter(augmented_recordings))}')
                return augmented_rejected
            for row in utils.shuffle(deepcopy(meta_df), random_state=rnd_state + permutation).iterrows():
                filename = row[1].filename
                patient = fname2patient(filename)
                recording = fname2rec(filename)
                recording_type = rec2rec_type(recording)
                source = fname2db(filename)

                if row[1].duration > 0:

                    if clf_type == 'validity':
                        fit_for_augmentation = 'ough' not in filename \
                                               and row[1].reject_id not in [1, 2, 3, 4, 5, 6]
                    else:
                        if recording not in phonemes:
                            fit_for_augmentation = True
                        elif phonemes[recording] <= mean_num_phonemes:
                            fit_for_augmentation = True
                        else:
                            fit_for_augmentation = False
                        # filename2db(filename) != 'Source3'

                    # if the recording is suitable
                    if recording_type in unique_recording_types \
                            and patient in accepted_patients \
                            and fit_for_augmentation \
                            and filename not in augmented_rejected \
                            and filename not in accepted \
                            and len(augmented_rejected) < len(accepted) \
                            and unique_recording_types[recording_type] <= mean_num_recording_types \
                            and augmented_patients.count(patient) <= accepted_patients.count(patient) + 2\
                            and augmented_sources[source] <= accepted_sources[source]*1.5:

                        augmented_patients.append(patient)
                        augmented_rejected.append(filename)
                        augmented_recordings.append(recording)
                        unique_recording_types[recording_type] += 1
                        augmented_sources[source] += 1
                        if recording in phonemes:
                            phonemes[recording] += 1

                    elif filename not in unfit_recordings:
                        unfit_recordings.append(filename)

        logging.info(
            f'rejected {str(len(augmented_rejected))} files in augmentation()'
            '\n\t\t\t\t\t\t\t\t\tfile types: '
            f'\n{str(Counter(augmented_recordings))}'
            '\n\t\t\t\t\t\t\t\t\tgeneral file types: '
            f'\n{str(unique_recording_types)}')

        return augmented_rejected

    def get_x_y(self, patient_names):
        # given X_y create either train or test data

        # lets pick only the des,fired patients for x
        x = deepcopy(self.embeddings_df)[
            [fname2patient(filename) in patient_names for filename in deepcopy(self.embeddings_df.filename)]]
        patients = [fname2patient(filename) for filename in x.filename]

        # get all the statistics on sources
        reject_sources = {}
        accept_sources = {}
        for row in x.iterrows():
            source = fname2db(row[1].filename)
            if source not in reject_sources:
                reject_sources[source] = 0
            if source not in accept_sources:
                accept_sources[source] = 0
            if row[1].rejected == 1:
                accept_sources[source] += 1
            elif row[1].rejected == 0:
                reject_sources[source] += 1

        if self.x_train.empty:
            mode = 'train'
        elif self.x_test.empty:
            mode = 'test'
        else:
            mode = 'mode broken'

        for source in accept_sources:
            logging.info(f'{accept_sources[source]} files were accepted '
                         f'in {mode} mode from {source} database')
        for source in reject_sources:
            logging.info(f'{reject_sources[source]} files were rejected '
                         f'in {mode} mode from {source} database')

        x.index = list(x.filename)
        x.pop('filename')

        # lets get rejected column out of x and into y
        y = np.array(x.pop('rejected'))

        for row in x.iterrows():
            source = 'broken_emb_path'

            for emb_path in self.embeddings_paths:
                if fname2db(row[0]) in emb_path:
                    source = emb_path

            if 'Source6' in source:
                file = np.load(source + row[0].split('_')[0] + '/' + row[0].split('_')[-1] + '.npy', encoding='ASCII')
            else:
                file = np.load(source + row[0] + '.npy', encoding='ASCII')

            if file.shape[0] != 1:
                if self.emb_type == 'median':
                    file = np.nanmedian(file, axis=0).reshape(1, -1)
                if self.emb_type == 'average':
                    file = np.average(file, axis=0).reshape(1, -1)
                if self.emb_type == 'middle':
                    mid_idx = int(file.shape[0] / 2)
                    file = file[mid_idx].reshape(1, -1)

            # original embeddings
            e_f = pd.DataFrame(file, columns=['feature_' + str(i + 1) for i in range(512)])
            # embeddings that will go to classifier
            x_f = x[x.index == row[0]].reset_index(drop=True)
            # whether they are the same thing or not
            true_db = x_f == e_f
            if not true_db.all().all():
                logging.info(
                    f'problem with embeddings of {row[0]} file in {mode} mode')

        return x, y, patients

    def get_train_test_id_split(self, augment, print_train_test=True):

        # find all rejected files and accepted coughs

        self.rnd_state += 1
        accepted, rejected, patients, recordings = self.get_accepted_rejected(self.meta_df,
                                                                              rnd_state=self.rnd_state,
                                                                              clf_type=self.clf_type)

        # add other filetypes as rejected
        if augment and len(rejected) < len(accepted):

            rejected = self.augmentation(self.meta_df, accepted, rejected, recordings,
                                         clf_type=self.clf_type, rnd_state=self.rnd_state)

        # prep full X, y dataframe
        self.meta_df = self.meta_df[
            [filename in rejected + accepted for filename in self.meta_df.filename]]
        self.meta_df['rejected'] = [int(filename in accepted) for filename in self.meta_df.filename]
        self.rnd_state += 1
        logging.info('meta dataframe:\n'
                     f'{pretty_print_df(self.meta_df, rnd_state=self.rnd_state, reject_col=True)}')
        self.get_all_embedding_dfs()
        self.embeddings_df['rejected'] = [int(filename in accepted) for filename in self.embeddings_df.filename]

        # make sure we have enough train and test material
        while sum(self.y_train) == len(self.y_train) or self.y_train == [] \
                or sum(self.y_test) == len(self.y_test) or self.y_test == []:
            # train_test_split, but don't mix patients between train and test
            unique_patients = np.unique(patients)
            source2patient_count = {}
            for patient in unique_patients:
                if fname2db(patient) in source2patient_count:
                    source2patient_count[fname2db(patient)].append(patient)
                else:
                    source2patient_count[fname2db(patient)] = [patient]

            test_patient_names = []
            for source in source2patient_count:
                source_patients = source2patient_count[source]
                k = int(len(source_patients) * 0.2)
                if k < 2:
                    k = 2
                self.rnd_state += 1
                random.seed(self.rnd_state)
                test_patient_names.extend(random.sample(list(source_patients), k=k))

            train_patient_names = list(set(unique_patients) - set(test_patient_names))

            # get train X, y
            self.x_train, self.y_train, self.train_patients = self.get_x_y(train_patient_names)
            # get test X, y
            self.x_test, self.y_test, self.test_patients = self.get_x_y(test_patient_names)

        if print_train_test:
            logging.info('we\'ve obtained train and test that look like:'
                         f'\n\t\t\t\t\t\t\t\t\taccepted train: {str(Counter(self.y_train)[1])}'
                         f'\n\t\t\t\t\t\t\t\t\trejected train: {str(Counter(self.y_train)[0])}'
                         f'\n\t\t\t\t\t\t\t\t\taccepted test: {str(Counter(self.y_test)[1])}'
                         f'\n\t\t\t\t\t\t\t\t\trejected test: {str(Counter(self.y_test)[0])}')

    def unite_train_test(self):
        self.x = pd.concat([self.x_train, self.x_test], sort=False)
        self.y = np.concatenate([self.y_train, self.y_test])
        self.patients = self.train_patients + self.test_patients

    def custom_cv(self, folds=10):
        # create cross-validation based on not mixing ids between training and validation

        ids = deepcopy(self.train_patients)

        kf = KFold(n_splits=folds, shuffle=True, random_state=self.rnd_state)
        all_ids = np.unique(ids)

        for train_index, valid_index in kf.split(all_ids):
            train_ids = all_ids[train_index]
            valid_ids = all_ids[valid_index]
            train_rec_ids = np.where(np.isin(ids, train_ids))[0]
            valid_rec_ids = np.where(np.isin(ids, valid_ids))[0]

            yield train_rec_ids, valid_rec_ids

    def train_clf(self, **kwargs):

        # train a range of classifiers to obtain the best one possible

        # create multiple classifiers
        attempts_in_permutation = 0
        while attempts_in_permutation < 100:
            if 'limited_permutations' in kwargs:
                attempts_in_permutation += 1
                self.rnd_state += 1
            self.parameters['random_state'] = [deepcopy(self.rnd_state)]
            scaler = StandardScaler()
            pca = PCA(svd_solver='randomized', random_state=deepcopy(self.rnd_state))
            svc_clf = svm.SVC(class_weight='balanced', probability=True, random_state=deepcopy(self.rnd_state))
            cv = self.custom_cv()
            gridsearch = GridSearchCV(svc_clf, deepcopy(self.parameters),
                                      cv=cv, scoring='roc_auc', n_jobs=-1, verbose=2)

            self.clf = make_pipeline(scaler, pca, gridsearch)

            # train them
            warnings.filterwarnings(action='ignore', category=DeprecationWarning)
            try:

                self.clf.fit(deepcopy(self.x_train), deepcopy(self.y_train))

                if 'limited_permutations' not in kwargs:
                    result_scaler = self.clf[0]
                    result_pca = self.clf[1]
                    result_clf = deepcopy(self.clf[-1].best_estimator_)
                    result_pipe = make_pipeline(result_scaler, result_pca, result_clf)
                    result_pipe = result_pipe.fit(self.x_train, self.y_train)
                    joblib.dump(result_pipe, self.clf_path)

                break
            except ValueError:
                logging.info('failed in training CoughClassifier')
                continue

    def find_important_features(self, clf4test):

        # get most important features for n_components of most important components
        most_important_features = []

        # get n_components of most important components
        for j, (feature_values, component_importance) in \
                enumerate(zip(clf4test[1].components_[:self.n_components],
                              clf4test[1].explained_variance_[:self.n_components])):
            # get the most important feature for each component
            feature_values_for_component = dict(zip(self.x_test.columns, feature_values))
            most_important_feature = max(feature_values_for_component, key=feature_values_for_component.get)
            most_important_features.append(most_important_feature)

        return most_important_features

    def check_sanity(self, params, permutations=10):
        # conduct a sanity check for 10 random label permutations

        insane_parameters = deepcopy(params)
        insane_parameters['C'] = [float(insane_parameters['C'])]
        insane_parameters['gamma'] = [float(insane_parameters['gamma'])]
        insane_parameters['kernel'] = tuple([insane_parameters['kernel']])
        insane_parameters['random_state'] = [insane_parameters['random_state']]

        insane_scores = []
        for permutation in range(permutations):
            insane_clf = CoughClassifier(self.meta_paths, self.embeddings_paths,
                                         parameters=insane_parameters,
                                         log_path=self.log_path, clf_path=self.clf_path, auc_path=self.auc_path,
                                         clf_type=self.clf_type, emb_type=self.emb_type)
            insane_clf.x_train = deepcopy(self.x_train)
            random.seed(self.rnd_state + permutation)
            insane_clf.y_train = random.sample(list(deepcopy(self.y_train)), len(self.y_train))
            insane_clf.train_patients = deepcopy(self.train_patients)
            insane_clf.rnd_state = deepcopy(self.rnd_state)
            insane_clf.x_test = deepcopy(self.x_test)
            insane_clf.y_test = deepcopy(self.y_test)
            try:
                insane_clf.train_clf(limited_permutations=True)
                insane_scores.append(insane_clf.clf.score(insane_clf.x_test, insane_clf.y_test))
            except (ValueError, NotFittedError):
                logging.info('failed in building insane classifier')
                continue

        if len(insane_scores) > 0:
            self.properties['sanity_results_average'] = sum(insane_scores) / len(insane_scores)
        else:
            self.properties['sanity_results_average'] = 'not available'

    def test_clf(self, sanity_check=False, pretty_print_params=False, printout=True, **kwargs):

        if 'clf' in kwargs:
            clf4test = kwargs['clf']
        else:
            clf4test = self.clf

        # test a classifier to make sure it works
        y_pred = clf4test.predict(self.x_test)
        y_probas = clf4test.predict_proba(self.x_test)

        # get important parameters to estimate classifier performance
        params = deepcopy(clf4test._final_estimator.best_params_)
        self.properties['most_important_parameters'] = params
        features = self.find_important_features(clf4test)
        self.properties['best_performing_features'] = features

        # get classifier results on test data
        results = {'accuracy': accuracy_score(self.y_test, y_pred),
                   'precision': precision_score(self.y_test, y_pred),
                   'recall': recall_score(self.y_test, y_pred),
                   'roc_auc': clf4test.score(self.x_test, self.y_test)}
        self.properties['different_metrics_results'] = results

        if pretty_print_params:
            # print all of the versions of classifiers
            logging.info('all classifier parameters are:'
                         f'\n{pretty_print_all_clf_options(clf4test._final_estimator)}')

        if printout:
            # print all of the important data
            logging.info(f'the best classifier parameters are:'
                         f'\n\t\t\t\t\t\t\t\t\t{str(params)}')
            for result in results:
                logging.info(": ".join([str(result), str(results[result])]))
            plot_roc(y_true=self.y_test, y_probas=y_probas)
            plt.savefig(self.auc_path)
            logging.info(f'{len(features)} most important features are:'
                         f'\n\t\t\t\t\t\t\t\t\t{", ".join(features)}')
            if sanity_check:
                self.check_sanity(params=params)
                logging.info('sanity check average results:'
                             f'\n\t\t\t\t\t\t\t\t\t{str(self.properties["sanity_results_average"])}')

    def get_classifier(self, augment=True, complex_split=False, sanity_check=False):

        # get all dataframes
        self.get_all_meta_dfs(complex_split=complex_split)
        # train and test classifier
        self.get_train_test_id_split(augment=augment)

        logging.info('the train_test parameters are'
                     f'\n\t\t\t\t\t\t\t\t\tx_train {str(self.x_train.shape)}'
                     f'\n\t\t\t\t\t\t\t\t\ty_train {str(len(self.y_train))}'
                     f'\n\t\t\t\t\t\t\t\t\ttrain_patients {str(len(np.unique(self.train_patients)))}'
                     f'\n\t\t\t\t\t\t\t\t\tx_test {str(self.x_test.shape)}'
                     f'\n\t\t\t\t\t\t\t\t\ty_test {str(len(self.y_test))}'
                     f'\n\t\t\t\t\t\t\t\t\ttest_patients {str(len(np.unique(self.test_patients)))}')

        self.train_clf()
        self.test_clf(sanity_check=sanity_check, printout=True)

        logging.info('Process finished with exit code: 0\n\n\n')

    def inner_cross_validation(self, augment=False, sanity_check=False, complex_split=False):

        # get all dataframes
        self.get_all_meta_dfs(complex_split=complex_split)

        # train and test classifier
        self.get_train_test_id_split(augment=augment, print_train_test=False)
        self.unite_train_test()

        self.train_clf()
        self.test_clf(sanity_check=sanity_check, printout=True)

        logging.info('Process finished with exit code: 0\n\n\n')

    def classifier_data_compat(self, augment=False, sanity_check=False, complex_split=False, **kwargs):

        # get all dataframes
        self.get_all_meta_dfs(complex_split=complex_split)

        # train and test classifier
        self.get_train_test_id_split(augment=augment, print_train_test=False)
        self.unite_train_test()

        if 'clf' in kwargs:
            clf4test = kwargs['clf']
        else:
            clf4test = self.clf

        self.test_clf(clf=clf4test, sanity_check=sanity_check, printout=True)

        logging.info('Process finished with exit code: 0\n\n\n')
