""" Module for fitting a Siamese Network model for pairwise comparisons.

This module is not imported automatically because Keras is a heavy import.
You must import it directly (for example, `import rankcentrality.siamese` or
`from rankcentrality import siamese`). In other words, simply importing
rankcentrality is not sufficient to access this module.
"""

from typing import Optional

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras import backend as K

from rankcentrality.types import Scores, Comparisons, ComparisonResults, Matrix


class SiameseNetRank:
    def __init__(
        self,
        n_items: int,
        comps: Comparisons,
        comp_results: ComparisonResults,
        item_features: Matrix,
    ):
        self.n_items = n_items
        self.comps = comps
        self.comp_results = comp_results
        self.X = item_features
        assert self.X.shape[0] == self.n_items
        self.n_features = self.X.shape[1]
        self.base_model = None
        self.model = None

    def _create_base_model(self, input_dim: int, hidden_dim: int = 20):
        self.base_model = Sequential()
        self.base_model.add(layers.Dense(hidden_dim, input_shape=(input_dim,)))
        self.base_model.add(layers.LeakyReLU())
        self.base_model.add(layers.Dropout(0.1))
        self.base_model.add(layers.Dense(hidden_dim))
        self.base_model.add(layers.LeakyReLU())
        self.base_model.add(layers.Dropout(0.1))
        self.base_model.add(layers.Dense(1))

    def _create_siam_model(self, input_dim: int):
        left_in = keras.Input((input_dim,))
        right_in = keras.Input((input_dim,))

        left_out, right_out = self.base_model(left_in), self.base_model(right_in)
        diff = layers.Subtract()([right_out, left_out])
        pred_layer = layers.Activation("sigmoid")(diff)
        self.model = keras.models.Model(
            inputs=[left_in, right_in], outputs=[pred_layer]
        )
        self.model.compile(
            "Adam", loss="binary_crossentropy", metrics=[keras.metrics.binary_accuracy],
        )

    def create_model(self, hidden_dim: int = 20):
        """Creates the SiameseNet model.

        If you are fitting the model manually (i.e., by using `self.model` or
        `self.fit()`), you must call this first. If you are using `run()` to
        fit the model, you need not call this method.

        Args:
            input_dim: the dimension of the input data (the number of
                features for each item in the dataset)
            hidden_dim: the number of dimensions in the hidden layer of the
                SiameseNet.
        """
        self._create_base_model(self.n_features, hidden_dim)
        self._create_siam_model(self.n_features)

    def fit(self, epochs: int, verbose: int = 0):
        """Fits a SiameseNet model for pairwise comparisons.

        Args:
            epochs: the number of epoches for which to train the model.
            verbose: verobsity level (see Keras documentation).
        """
        X_left = self.X[self.comps[:, 0]]
        X_right = self.X[self.comps[:, 1]]
        y = self.comp_results
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            "loss", factor=0.5, min_lr=0.0001, patience=3, verbose=verbose
        )
        self.model.fit(
            [X_left, X_right], y, epochs=epochs, callbacks=[reduce_lr], verbose=verbose
        )

    def predict(self, X: Optional[Matrix] = None) -> Scores:
        """Returns the decision function evaluated on X

        Args:
            X: a matrix of shape (n_items, n_features). If not specified,
                defaults to the matrix passed to the constructor.

        Returns:
            Scores for each item. Note that the output scores are *not* BTL
            scores. They are simply the values of the decision function used
            by the model to predict which item a pair will be prefered in a
            comparison.
        
        """
        X = self.X if X is None else X
        # Number of rows need not be self.n_items if predicting for items
        # outside the training data.
        assert X.shape[1] == self.n_features
        return self.base_model.predict(X).flatten().astype(np.float64)

    def run(self, epochs: int = 10, verbose: int = 0) -> Scores:
        """Fits a SiameseNet model and returns their predicted scores.

        Args:
            epochs: the number of epoches for which to train the model.
            verbose: verobsity level (see Keras documentation).
        
        Returns:
            Scores for each item. Note that the output scores are *not* BTL
            scores. They are simply the values of the decision function used
            by the model to predict which item a pair will be prefered in a
            comparison.
        """
        self.create_model()
        self.fit(epochs, verbose)
        return self.predict()
