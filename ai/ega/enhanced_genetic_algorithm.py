import numpy as np


class EnhancedGeneticAlgorithm:

    def __init__(
        self,
        n_features,
        population_size=16,
        generations=15,
        mutation_rate=0.02,
        crossover_rate=0.8,
        max_features=512,
        random_state=42
    ):

        self.n_features = n_features
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.max_features = min(max_features, n_features)

        self.rng = np.random.default_rng(random_state)

        self.best_mask_ = None
        self.best_score_ = None

    # --------------------------------------------------
    # CREATE INITIAL POPULATION
    # --------------------------------------------------

    def _population(self):

        population = []

        for _ in range(self.population_size):

            # Start with a sparse chromosome
            chromosome = np.zeros(
                self.n_features,
                dtype=np.int8
            )

            number_of_features = self.rng.integers(
                max(1, min(20, self.max_features)),
                self.max_features + 1
            )

            selected = self.rng.choice(
                self.n_features,
                size=number_of_features,
                replace=False
            )

            chromosome[selected] = 1

            population.append(chromosome)

        return np.asarray(population)

    # --------------------------------------------------
    # FITNESS FUNCTION
    # --------------------------------------------------

    def _fitness(
        self,
        mask,
        X_train,
        y_train,
        X_val,
        y_val
    ):

        selected_indices = np.flatnonzero(mask)

        number_selected = len(selected_indices)

        if number_selected == 0:
            return -1

        from sklearn.linear_model import LogisticRegression

        try:

            model = LogisticRegression(
                max_iter=300,
                solver="lbfgs"
            )

            model.fit(
                X_train[:, selected_indices],
                y_train
            )

            accuracy = model.score(
                X_val[:, selected_indices],
                y_val
            )

            # Small penalty for using too many features
            feature_penalty = (
                0.001 *
                number_selected /
                self.n_features
            )

            fitness = accuracy - feature_penalty

            return fitness

        except Exception:

            return -1

    # --------------------------------------------------
    # FIT EGA
    # --------------------------------------------------

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val
    ):

        population = self._population()

        best_score = -np.inf
        best_mask = None

        for generation in range(self.generations):

            scores = []

            for chromosome in population:

                score = self._fitness(
                    chromosome,
                    X_train,
                    y_train,
                    X_val,
                    y_val
                )

                scores.append(score)

            scores = np.asarray(scores)

            best_index = np.argmax(scores)

            generation_best_score = scores[best_index]

            # Update global best
            if generation_best_score > best_score:

                best_score = float(
                    generation_best_score
                )

                best_mask = population[
                    best_index
                ].copy()

            # Safety check
            if best_mask is None:

                print(
                    f"Generation "
                    f"{generation + 1}/"
                    f"{self.generations}: "
                    f"No valid solution yet"
                )

            else:

                print(
                    f"Generation "
                    f"{generation + 1}/"
                    f"{self.generations}: "
                    f"fitness={best_score:.5f}, "
                    f"features={best_mask.sum()}"
                )

            # --------------------------------------------------
            # ELITE SELECTION
            # --------------------------------------------------

            elite_count = max(
                2,
                self.population_size // 4
            )

            elite_indices = np.argsort(
                scores
            )[::-1][:elite_count]

            elite = population[
                elite_indices
            ]

            # --------------------------------------------------
            # CREATE NEXT GENERATION
            # --------------------------------------------------

            children = [
                elite[0].copy()
            ]

            while len(children) < self.population_size:

                parent1 = elite[
                    self.rng.integers(
                        len(elite)
                    )
                ]

                parent2 = elite[
                    self.rng.integers(
                        len(elite)
                    )
                ]

                child = parent1.copy()

                # Crossover
                if (
                    self.rng.random()
                    <
                    self.crossover_rate
                ):

                    point = self.rng.integers(
                        1,
                        self.n_features
                    )

                    child[point:] = (
                        parent2[point:]
                    )

                # Mutation
                mutation = (
                    self.rng.random(
                        self.n_features
                    )
                    <
                    self.mutation_rate
                )

                child[mutation] = (
                    1 -
                    child[mutation]
                )

                # Ensure at least one feature
                if child.sum() == 0:

                    child[
                        self.rng.integers(
                            self.n_features
                        )
                    ] = 1

                # Limit number of features
                if child.sum() > self.max_features:

                    selected = np.flatnonzero(
                        child
                    )

                    remove_count = (
                        len(selected)
                        -
                        self.max_features
                    )

                    remove_indices = self.rng.choice(
                        selected,
                        size=remove_count,
                        replace=False
                    )

                    child[
                        remove_indices
                    ] = 0

                children.append(child)

            population = np.asarray(children)

        # --------------------------------------------------
        # FINAL RESULT
        # --------------------------------------------------

        if best_mask is None:

            raise RuntimeError(
                "EGA could not find a valid feature subset. "
                "Check train/validation features and labels."
            )

        self.best_mask_ = best_mask.astype(bool)

        self.best_score_ = best_score

        print("\n======================================")
        print("EGA FEATURE SELECTION COMPLETED")
        print("======================================")
        print(
            "Total features :",
            self.n_features
        )
        print(
            "Selected features :",
            self.best_mask_.sum()
        )
        print(
            "Best fitness :",
            round(self.best_score_, 5)
        )

        return self

    # --------------------------------------------------
    # TRANSFORM
    # --------------------------------------------------

    def transform(self, X):

        if self.best_mask_ is None:

            raise RuntimeError(
                "EGA has not been fitted yet."
            )

        return X[
            :,
            self.best_mask_
        ]

    # --------------------------------------------------
    # GET SELECTED FEATURES
    # --------------------------------------------------

    def get_support(self):

        if self.best_mask_ is None:

            raise RuntimeError(
                "EGA has not been fitted yet."
            )

        return self.best_mask_