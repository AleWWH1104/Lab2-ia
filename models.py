import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    """Regresión Logística implementada con gradiente descendente."""

    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None
        self.loss_history = []

    @staticmethod
    def sigmoid(z):
        """Función sigmoide: g(z) = 1 / (1 + e^(-z))"""
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        """Predicción de probabilidades: y_hat = g(X·w + b)"""
        z = np.dot(X, self.w) + self.b
        return self.sigmoid(z)

    def log_loss(self, y_true, y_pred):
        """Log Loss (Binary Cross-Entropy)"""
        m = len(y_true)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def fit(self, X, y):
        """Entrena el modelo usando gradient descent."""
        m, n = X.shape

        # Inicializar pesos y sesgo
        self.w = np.zeros(n)
        self.b = 0
        self.loss_history = []

        # Entrenamiento
        for epoch in range(self.epochs):
            # Forward pass
            y_pred = self.predict_proba(X)

            # Calcular loss
            loss = self.log_loss(y, y_pred)
            self.loss_history.append(loss)

            # Calcular gradientes
            dw = (1/m) * np.dot(X.T, (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)

            # Actualizar parámetros
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

        print(f"\nRegresión Logística entrenada")
        print(f"Loss final: {self.loss_history[-1]:.4f}")

    def predict(self, X):
        """Predicción de clases (0 o 1)."""
        return (self.predict_proba(X) >= 0.5).astype(int)

    def plot_loss_curve(self, output_path='outputs/loss_curve.png'):
        """Grafica la curva de pérdida."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, linewidth=2)
        plt.xlabel('Época', fontsize=12)
        plt.ylabel('Log Loss', fontsize=12)
        plt.title('Curva de Aprendizaje - Regresión Logística', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_decision_boundary(self, X_train, y_train, feature_names,
                               output_path='outputs/decision_threshold.png'):
        """Grafica la frontera de decisión."""
        plt.figure(figsize=(10, 8))

        # Scatter de puntos
        colors = ['blue' if y == 0 else 'red' for y in y_train]
        plt.scatter(X_train[:, 0], X_train[:, 1], c=colors, alpha=0.5,
                   edgecolors='k', linewidth=0.5)

        # Dibujar la línea de decisión
        # La frontera es donde: w[0]*x1 + w[1]*x2 + b = 0
        # Despejando: x2 = -(w[0]*x1 + b) / w[1]
        x1_min, x1_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
        x1_line = np.linspace(x1_min, x1_max, 100)
        x2_line = -(self.w[0] * x1_line + self.b) / self.w[1]

        plt.plot(x1_line, x2_line, 'g-', linewidth=3, label='Frontera de Decisión')
        plt.xlabel(feature_names[0], fontsize=12)
        plt.ylabel(feature_names[1], fontsize=12)
        plt.title('Frontera de Decisión - Regresión Logística', fontsize=14, fontweight='bold')
        plt.legend(['Frontera', 'Legitimate', 'Phishing'])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

class KNN:
    """K-Nearest Neighbors"""

    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    @staticmethod
    def euclidean_distance(x1, x2):
        """Calcula la distancia euclidiana entre dos puntos."""
        return np.sqrt(np.sum((x1 - x2)**2))

    def fit(self, X_train, y_train):
        """Guarda los datos de entrenamiento."""
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """Predice las clases para los datos de prueba usando vectorización numpy."""
        predictions = []

        for test_point in X_test:
            # Calcular distancias a todos los puntos de training (vectorizado)
            distances = np.sqrt(np.sum((self.X_train - test_point)**2, axis=1))

            # Obtener índices de los k vecinos más cercanos
            k_indices = np.argsort(distances)[:self.k]

            # Obtener las etiquetas de los k vecinos más cercanos
            k_labels = self.y_train[k_indices]

            # Votación mayoritaria
            prediction = 1 if np.sum(k_labels) > self.k/2 else 0
            predictions.append(prediction)

        return np.array(predictions)

    def plot_decision_boundary(self, X_train, y_train, feature_names,
                               output_path='outputs/decision_boundary_knn.png'):
        """Grafica el mapa de decisión de KNN."""

        plt.figure(figsize=(10, 8))

        # Crear malla de puntos para el mapa de color
        x1_min, x1_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        x2_min, x2_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        h = 0.05  # Resolución de la malla

        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                               np.arange(x2_min, x2_max, h))

        # Predecir para cada punto de la malla
        Z = self.predict(np.c_[xx1.ravel(), xx2.ravel()])
        Z = Z.reshape(xx1.shape)

        # Dibujar el mapa de color
        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap='coolwarm')

        # Scatter de puntos
        colors = ['blue' if y == 0 else 'red' for y in y_train]
        plt.scatter(X_train[:, 0], X_train[:, 1], c=colors, alpha=0.7,
                   edgecolors='k', linewidth=0.5)

        plt.xlabel(feature_names[0], fontsize=12)
        plt.ylabel(feature_names[1], fontsize=12)
        plt.title(f'Decision Boundary - KNN (k={self.k})', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
