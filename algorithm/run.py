from basic_classifiers import CoughClassifier
import MainConfig as MainCfg

def randomized_run(clf_type='regular'):

    for rnd_state in [19, 39, 132, 314, 516]:
        log_path = MainCfg.LOG_RND_PATH + '_rnd' + '.log'
        clf_path = MainCfg.CLF_RND_PATH + '_rnd_' + str(rnd_state) + '.pkl'
        auc_path = MainCfg.AUC_RND_PATH + '_rnd_' + str(rnd_state) + '.png'

        meta_paths = [MainCfg.S1_META_PATH, MainCfg.S2_META_PATH,
                      MainCfg.S3_META_PATH, MainCfg.S4_META_PATH,
                      MainCfg.S5_META_PATH, MainCfg.S6_META_PATH]
        embeddings_paths = [MainCfg.S1_EMB_PATH, MainCfg.S2_EMB_PATH,
                            MainCfg.S3_EMB_PATH, MainCfg.S4_EMB_PATH,
                            MainCfg.S5_EMB_PATH, MainCfg.S6_EMB_PATH]

        rnd_clf = CoughClassifier(meta_paths, embeddings_paths, log_path=log_path,
                                  clf_path=clf_path, auc_path=auc_path, clf_type=clf_type, rnd_state=rnd_state)
        rnd_clf.get_classifier(augment=True, complex_split=True, sanity_check=False)


def randomized_5emb_run(clf_type='regular'):

    for rnd_state in [12, 39, 46, 103, 413, 789, 1013, 3456, 5700, 9412]:
        log_path = MainCfg.LOG_RND_PATH + '_rnd' + '.log'
        clf_path = MainCfg.CLF_RND_PATH + '_rnd_' + str(rnd_state) + '.pkl'
        auc_path = MainCfg.AUC_RND_PATH + '_rnd_' + str(rnd_state) + '.png'

        meta_paths = [MainCfg.S1_META_PATH, MainCfg.S2_META_PATH,
                      MainCfg.S3_META_PATH, MainCfg.S4_META_PATH,
                      MainCfg.S5_META_PATH, MainCfg.S6_META_PATH]
        embeddings_paths = [MainCfg.S1_EMB5_PATH, MainCfg.S2_EMB5_PATH,
                            MainCfg.S3_EMB5_PATH, MainCfg.S4_EMB5_PATH,
                            MainCfg.S5_EMB5_PATH, MainCfg.S6_EMB5_PATH]

        rnd_clf = CoughClassifier(meta_paths, embeddings_paths, log_path=log_path,
                                  clf_path=clf_path, auc_path=auc_path, clf_type=clf_type, rnd_state=rnd_state)
        rnd_clf.get_classifier(augment=True, complex_split=True, sanity_check=True)

def final_run(clf_type='regular'):

    rnd_state = 516
    log_path = MainCfg.LOG_RND_PATH + '_rnd' + '.log'
    clf_path = MainCfg.CLF_RND_PATH + '_rnd_' + str(rnd_state) + '.pkl'
    auc_path = MainCfg.AUC_RND_PATH + '_rnd_' + str(rnd_state) + '.png'

    meta_paths = [MainCfg.S1_META_PATH, MainCfg.S2_META_PATH,
                      MainCfg.S3_META_PATH, MainCfg.S4_META_PATH,
                      MainCfg.S5_META_PATH, MainCfg.S6_META_PATH]
        embeddings_paths = [MainCfg.S1_EMB5_PATH, MainCfg.S2_EMB5_PATH,
                            MainCfg.S3_EMB5_PATH, MainCfg.S4_EMB5_PATH,
                            MainCfg.S5_EMB5_PATH, MainCfg.S6_EMB5_PATH]

    rnd_clf = CoughClassifier(meta_paths, embeddings_paths, log_path=log_path,
                              clf_path=clf_path, auc_path=auc_path, clf_type=clf_type, rnd_state=rnd_state)
    rnd_clf.get_classifier(augment=True, complex_split=True, sanity_check=False)


if __name__ == "__main__":
    final_run()
