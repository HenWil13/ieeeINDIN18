import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import PCA

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from scipy import sparse

np.random.seed(42)


class DataPreparation(object):


    def __init__(self, ds_path, num_attribs_feat=[None], cat_attribs_feat=[None], num_attribs_lab=[None]):
        """ 
        
        """

        self._ds_path = ds_path

        # features and labels as np
        self._features = pd.read_csv(str(self._ds_path+'/features.csv'), index_col=0)
        self._labels = pd.read_csv(str(self._ds_path+'/labels.csv'), index_col=0)

        print("Make sure that ds_path+/features.csv as well as ds_path+/labels.csv are in the defined format.\n"
              "Especially check that labels don't contain any time stamps or similar other than the index column.\n"
              "Scaling of labels particularly makes sense for models with auto-regressive character. \n"
              "Labels other than those defined in num_attrib_labs will be dropped automatically when labels are scaled")

        assert len(self._features) == len(self._labels), "Label and feature length don't match"

        self._num_attribs_feat = num_attribs_feat
        self._cat_attribs_feat = cat_attribs_feat

        self._num_attribs_lab = num_attribs_lab

        self._n_instances, self._n_features = self._features.shape
        self._n_labels = self._labels.shape[1]

        self._prep_pipeline_feat = None
        self._prepared_features = np.array(None)
        self._prep_pipeline_lab = None
        self._prepared_labels = np.array(None)
        self._dim_reducer = None


    def create_fit_pipeline_feat(self, scaler=None):
        """ 

        """

        if scaler == None:
            sc = StandardScaler()
        if scaler == "min-max":
            sc = MinMaxScaler()
        if scaler == "robust":
            sc = RobustScaler()
        if scaler == "normalizer":
            sc = Normalizer()

        class NumDataFrameSelector(BaseEstimator, TransformerMixin):
            def __init__(self, attribute_names):
                self.attribute_names = attribute_names

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X[self.attribute_names].values

        class CatDataFrameSelector(BaseEstimator, TransformerMixin):
            def __init__(self, attribute_names):
                self.attribute_names = attribute_names

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X[self.attribute_names].values.reshape(-1, 1)

        transformer_list = []

        num_pipeline = Pipeline([
            ('selector', NumDataFrameSelector(self._num_attribs_feat)),
            ('std_scaler', sc),
        ])

        transformer_list.append(("num_pipeline", num_pipeline))

        for i in range(len(self._cat_attribs_feat)):
            cat_pipeline = Pipeline([
                ('selector', CatDataFrameSelector(self._cat_attribs_feat[i])),
                ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")), ])
            pipeline_name = "cat_pipeline_{}".format(i)
            transformer_list.append((pipeline_name, cat_pipeline))

        self._prep_pipeline_feat = FeatureUnion(transformer_list)

        self._prepared_features = self._prep_pipeline_feat.fit_transform(self._features)

    def create_fit_pipeline_lab(self, scaler=None):
        """
        
        """
        if scaler == None:
            sc = StandardScaler()
        if scaler == "min-max":
            sc = MinMaxScaler()
        if scaler == "robust":
            sc = RobustScaler()
        if scaler == "normalizer":
            sc = Normalizer()

        class NumDataFrameSelector(BaseEstimator, TransformerMixin):
            def __init__(self, attribute_names):
                self.attribute_names = attribute_names

            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X[self.attribute_names].values

        transformer_list = []

        self._prep_pipeline_lab = Pipeline([
            ('selector', NumDataFrameSelector(self._num_attribs_lab)),
            ('scaler', sc),
        ])

        self._prepared_labels = self._prep_pipeline_lab.fit_transform(self._labels)

    def fit_pca(self, explained_variance, apply=False):
        """ """

        pca = PCA()
        pca.fit(self._prepared_features)
        cumsum = pca.explained_variance_ratio_.cumsum()
        dims = np.argmax(cumsum >= explained_variance) + 1

        if apply:
            pca = PCA(n_components=dims, random_state=42)
            pca.fit(self._prepared_features)  # must be performed on entire DS!!!
            print("Reducing dimensions to {}.  Explained Variance: {:.5f}. \n"
                  "Dimensions will be reduced during Training and Inference".format(dims,
                                                                            np.sum(pca.explained_variance_ratio_)))
            return pca

        else:
            pca = PCA(n_components=dims, random_state=42)
            pca.fit(self._prepared_features)  # must be performed on entire DS!!!
            print("Reducing dimensions to {}.  Explained Variance: {:.4f}. \n"
                  "If dimensions should be reduced set apply to True".format(dims,
                                                                             np.sum(pca.explained_variance_ratio_)))
            return None

    def reduce_dimensions(self, explained_variance=0.95, method='PCA', apply=False):
        """
        performs PCA or t-SNE analysis and reduces features to threshold of explained variance
        should be called after the preparation pipeline is fitted as one hot vectorized categorical attribs and scaled
        attribs should be evaluated for reduction
        self._prepared_features contains the features the pipeline was fitted on

            args:
        threshold: threshold of explained variance of the features on the label(s)
        method: PCA or t-SNE
        apply: if set to true features to reach threshold will be stored in self._reduced_features. during next batch
               method these features will be reselected before being fed into the model
        """

        if not self._prepared_features.any():
            raise ValueError("Transform data before reducing features using create_fit_pipeline_feat method")

        implemented_methods = ["PCA"]

        if method == 'PCA':
            self._dim_reducer = self.fit_pca(apply=apply, explained_variance=explained_variance)

        elif method not in implemented_methods:
            raise NotImplementedError("Method {} unknown or not implemented".format(method))

    def windows(self, data, window_size, overlap=1):
        start = 0
        while start < len(data):
            yield int(start), int(start + window_size)
            start += (window_size / overlap)

    def create_seq_data(self, features, labels, seq_len):

        assert features.shape[0] == labels.shape[0], "feature and label size don't match"

        n_features = features.shape[1]
        n_labels = labels.shape[1]

        n_instances = int(np.ceil(features.shape[0] / seq_len)) - 1

        features = features.reshape(-1, n_features)
        labels = labels.reshape(-1, n_labels)

        features_seq = np.zeros((n_instances, seq_len, n_features))
        labels_seq = np.zeros((n_instances, seq_len, n_labels))

        i = 0
        for (start, end) in self.windows(features, window_size=seq_len):
            f_seq = features[start:end]
            l_seq = labels[start:end]

            if (len(features[start:end]) < (seq_len)):
                diff = seq_len - len(features[start:end])  # Differenz für padding

                f_seq = np.pad(f_seq, [(0, diff), (0, 0)], mode='constant', constant_values=0)
                l_seq = np.pad(l_seq, [(0, diff), (0, 0)], mode='constant', constant_values=0)

            f_seq = f_seq.reshape(1, seq_len, n_features)
            l_seq = l_seq.reshape(1, seq_len, n_labels)

            # IndexError exception in case last sequence out of n_instances range. n_instances floored so that
            # only complete and not 0-padded sequences are built
            try:
                features_seq[i] = f_seq
            except IndexError:
                continue
            try:
                labels_seq[i] = l_seq
            except IndexError:
                continue
            i += 1

        return features_seq, labels_seq

    def split_data(self, model, sequence_length, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """
        
        
        todo: - evaluate first sequentializing ds and then splitting randomly using sklearn train_test_split
                instead of fixed ratio split along index(==time)
              - pass transformer pipeline to DataSet class and transform features and labels within
        """
        if np.sum([train_ratio, val_ratio, test_ratio]) != 1:
            raise ValueError("Sum of ratios does not equal 1")

        if model=="seq2seq":
            sequence_length += 1  # for later shifting of features against labels

        # transform data if selected in data prep
        if self._prep_pipeline_feat:
            f = self._prep_pipeline_feat.transform(self._features)
        else:
            f = self._features
        if self._dim_reducer:
            f = self._dim_reducer.transform(f)
        # labels can also be scaled, especially for seq2seq models
        # pipeline is passed to DataSet class and can be recalled within model to
        # reverse scaling of labels using pipeline.inverse_transform(X)
        if self._prep_pipeline_lab:
            l = self._prep_pipeline_lab.transform(self._labels)
        else:
            l = self._labels

        # split set
        split_train = np.floor(self._n_instances * train_ratio).astype(int)
        split_val = split_train + np.floor(self._n_instances * val_ratio).astype(int)

        train_f = np.array(f[:split_train])
        train_l = np.array(l[:split_train])

        val_f = np.array(f[split_train:split_val])
        val_l = np.array(l[split_train:split_val])

        test_f = np.array(f[split_val:])
        test_l = np.array(l[split_val:])

        # sequencialize sets
        # type error if none is passed due to one of the ratios being 0
        try:
            train_f, train_l = self.create_seq_data(train_f, train_l, sequence_length)
        except TypeError:
            pass
        try:
            val_f, val_l = self.create_seq_data(val_f, val_l, sequence_length)
        except TypeError:
            pass
        try:
            test_f, test_l = self.create_seq_data(test_f, test_l, sequence_length)
        except TypeError:
            pass

        train = DataSet(features=train_f, labels=train_l, model=model,
                        prep_pipeline_feat=self._prep_pipeline_feat, prep_pipeline_lab=self._prep_pipeline_lab,
                        dim_reducer=self._dim_reducer)
        val = DataSet(features=val_f, labels=val_l, model=model,
                      prep_pipeline_feat=self._prep_pipeline_feat, prep_pipeline_lab=self._prep_pipeline_lab,
                      dim_reducer=self._dim_reducer)
        test = DataSet(features=test_f, labels=test_l, model=model,
                       prep_pipeline_feat=self._prep_pipeline_feat, prep_pipeline_lab=self._prep_pipeline_lab,
                       dim_reducer=self._dim_reducer)

        return train, val, test


class DataSet(object):
    """
    
    """

    def __init__(self, features, labels, model, prep_pipeline_feat=None, prep_pipeline_lab=None, dim_reducer=None):
        """ 
        
        """

        self._features = features
        self._labels = labels

        self.n_sequences, self.sequence_length, self.n_features = self._features.shape
        self.n_labels = self._labels.shape[2]

        implemented_batch_methods = ["seq2seq"]
        self.model = model

        self.n_iterations = None
        self._index_in_epoch = None

        if self.model == "seq2seq":
            # set correct next_batch methoc
            self.next_batch = self.next_batch_seq2seq
            # convert features and labels to list for shift iteration, necessary?
            f_ls = self._features.tolist()
            l_ls = self._labels.tolist()
            # shift features and labels against each other for seq2seq model
            self.s2s_feat = np.array([f_ls[seq][1:] for seq in range(self.n_sequences)])
            self.s2s_shifted_lab = np.array([l_ls[seq][:-1] for seq in range(self.n_sequences)])
            self.s2s_tar = np.array([l_ls[seq][1:] for seq in range(self.n_sequences)])

        elif self.model not in implemented_batch_methods:
            raise NotImplementedError("{} model unknown or next_batch_method not implemented".format(self.model))

        self.prep_pipeline_feat = prep_pipeline_feat
        self.prep_pipeline_lab = prep_pipeline_lab
        if self.prep_pipeline_lab:
            self.lab_scaler = self.prep_pipeline_lab.named_steps['scaler']
        self.dim_reducer = dim_reducer

        self.reset_epoch()

    def reset_epoch(self):
        self._index_in_epoch = 0

    def get_iterations(self, batch_size):
        self.n_iterations = int(np.ceil(self.n_sequences/batch_size))  # ceil -> last batch is incomplete
        return self.n_iterations

    def next_batch_seq2seq(self, enc_len, dec_len, batch_size, shuffle=True):
        """
        
        """
        if enc_len + dec_len != self.s2s_feat.shape[1]:
            raise ValueError("sequence length {} in data set not of same length as dec_len + enc_len".format(
                self.s2s_feat.shape[1]))

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        end = self._index_in_epoch

        enc_seqlen = np.ones(batch_size) * enc_len
        dec_seqlen = np.ones(batch_size) * dec_len

        if shuffle and start == 0:
            perm = np.random.permutation(self.n_sequences)
            # shuffled features, labels and targets
            f = self.s2s_feat[perm]
            s_l = self.s2s_shifted_lab[perm]
            t = self.s2s_tar[perm]

            # store shuffled data in class for next iteration or reshuffling at end of epoch/beginning of next epoch
            self.s2s_feat = f
            self.s2s_shifted_lab = s_l
            self.s2s_tar = t

        else:
            f = self.s2s_feat
            s_l = self.s2s_shifted_lab
            t = self.s2s_tar

        if start + batch_size > self.n_sequences:
            # entire data set will be completely run through within this iteration
            # marks the end of the epoch
            # extract remaining sequences out of data set
            remaining_sequences = self.n_sequences - start
            f_rem = f[start:self.n_sequences]
            s_l_rem = s_l[start:self.n_sequences]
            t_rem = t[start:self.n_sequences]

            # reshuffle data set if shuffle is set to true
            if shuffle:
                perm = np.random.permutation(self.n_sequences)
                # shuffled features, labels and targets
                f = self.s2s_feat[perm]
                s_l = self.s2s_shifted_lab[perm]
                t = self.s2s_tar[perm]

                # store shuffled data in class for next iteration or reshuffling at end of epoch/beginning of next epoch
                self.s2s_feat = f
                self.s2s_shifted_lab = s_l
                self.s2s_tar = t

            # start with (newly shuffled) data set and fill up remaining sequences until batch_size is complete
            start = 0
            self._index_in_epoch = batch_size - remaining_sequences
            end = self._index_in_epoch

            f = f[start:end]
            s_l = s_l[start:end]
            t = t[start:end]

            f = np.concatenate((f_rem, f),axis=0)
            s_l = np.concatenate((s_l_rem, s_l), axis=0)
            t = np.concatenate((t_rem, t), axis=0)

        else:
            # extract batch sequences
            f = f[start:end]
            s_l = s_l[start:end]
            t = t[start:end]

        # divide batch sequences in encoder and decoder
        # encoder
        enc_f = f[:, :enc_len, :]
        enc_s_l = s_l[:, :enc_len, :]
        enc_inp = np.concatenate((enc_f, enc_s_l), axis=2)
        enc_tar = t[:, :enc_len, :]

        # decoder
        dec_f = f[:, enc_len:, :]
        dec_s_l = s_l[:, enc_len:, :]
        dec_tr_inp = np.concatenate((dec_f, dec_s_l), axis=2)
        dec_inf_inp = dec_f
        dec_tar = t[:, enc_len:, :]

        return enc_inp, enc_seqlen, dec_tr_inp, dec_inf_inp, dec_seqlen, enc_tar, dec_tar


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features as a numeric array.
    The input to this transformer should be a matrix of integers or strings,
    denoting the values taken on by categorical (discrete) features.
    The features can be encoded using a one-hot aka one-of-K scheme
    (``encoding='onehot'``, the default) or converted to ordinal integers
    (``encoding='ordinal'``).
    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.
    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.
    Parameters
    ----------
    encoding : str, 'onehot', 'onehot-dense' or 'ordinal'
        The type of encoding to use (default is 'onehot'):
        - 'onehot': encode the features using a one-hot aka one-of-K scheme
          (or also called 'dummy' encoding). This creates a binary column for
          each category and returns a sparse matrix.
        - 'onehot-dense': the same as 'onehot' but returns a dense array
          instead of a sparse matrix.
        - 'ordinal': encode the features as ordinal integers. This results in
          a single column of integers (0 to n_categories - 1) per feature.
    categories : 'auto' or a list of lists/arrays of values.
        Categories (unique values) per feature:
        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories are sorted before encoding the data
          (used categories can be found in the ``categories_`` attribute).
    dtype : number type, default np.float64
        Desired dtype of output.
    handle_unknown : 'error' (default) or 'ignore'
        Whether to raise an error or ignore if a unknown categorical feature is
        present during transform (default is to raise). When this is parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros.
        Ignoring unknown categories is not supported for
        ``encoding='ordinal'``.
    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting. When
        categories were specified manually, this holds the sorted categories
        (in order corresponding with output of `transform`).
    Examples
    --------
    Given a dataset with three features and two samples, we let the encoder
    find the maximum value per feature and transform the data to a binary
    one-hot encoding.
    >> from sklearn.preprocessing import CategoricalEncoder
    >> enc = CategoricalEncoder(handle_unknown='ignore')
    >> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
    ... # doctest: +ELLIPSIS
    CategoricalEncoder(categories='auto', dtype=<... 'numpy.float64'>,
              encoding='onehot', handle_unknown='ignore')
    >> enc.transform([[0, 1, 1], [1, 0, 4]]).toarray()
    array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]])
    See also
    --------
    sklearn.preprocessing.OneHotEncoder : performs a one-hot encoding of
      integer ordinal features. The ``OneHotEncoder assumes`` that input
      features take on values in the range ``[0, max(feature)]`` instead of
      using the unique values.
    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
      encoding of dictionary items or strings.
    """

    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Fit the CategoricalEncoder to X.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_feature]
            The data to determine the categories of each feature.
        Returns
        -------
        self
        """

        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        """Transform X using one-hot encoding.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.
        Returns
        -------
        X_out : sparse matrix or a 2-d array
            Transformed input.
        """
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out