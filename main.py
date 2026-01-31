import numpy as np
from data_processor import DataProcessor
from models import LogisticRegression, KNN


def main():

    # Preprocesamiento de datos
    print("\nPreprocesamiento de datos")
    print("-" * 20)

    processor = DataProcessor(
        dataset_path='dataset_phishing.csv',
        output_dir='outputs'
    )

    X_train, X_test, y_train, y_test, feature_names = processor.process_all()


    #Regresión Logística
    print("\nRegresión Logística")
    print("-" * 20)

    log_reg = LogisticRegression(learning_rate=0.1, epochs=1000)
    log_reg.fit(X_train, y_train)

    # Predicciones
    y_pred_train_lr = log_reg.predict(X_train)
    y_pred_test_lr = log_reg.predict(X_test)

    # Accuracy
    train_acc_lr = np.mean(y_pred_train_lr == y_train)
    test_acc_lr = np.mean(y_pred_test_lr == y_test)

    print(f"\nAccuracy - Regresión Logística:")
    print(f"  Training: {train_acc_lr:.4f}")
    print(f"  Testing: {test_acc_lr:.4f}")

    # Gráficas
    log_reg.plot_loss_curve()
    log_reg.plot_decision_boundary(X_train, y_train, feature_names)

    
    # KNN
    print("\nK-Nearest Neighbors")
    print("-" * 20)

    knn = KNN(k=3)
    knn.fit(X_train, y_train)

    # Predicciones
    y_pred_train_knn = knn.predict(X_train)
    y_pred_test_knn = knn.predict(X_test)

    # Accuracy
    train_acc_knn = np.mean(y_pred_train_knn == y_train)
    test_acc_knn = np.mean(y_pred_test_knn == y_test)

    print(f"\nAccuracy - KNN (k={knn.k}):")
    print(f"  Training: {train_acc_knn:.4f}")
    print(f"  Testing: {test_acc_knn:.4f}")

    # Gráfica
    knn.plot_decision_boundary(X_train, y_train, feature_names)

    # Resumen
    print("\n" + "-" * 20)
    print("Resumen de Resultados")
    print("-" * 20)
    print(f"\nFeatures utilizadas:")
    print(f"1. {feature_names[0]}")
    print(f"2. {feature_names[1]}")

    print(f"\nComparación de modelos (Test Accuracy):")
    print(f"  Regresión Logística: {test_acc_lr:.4f}")
    print(f"  KNN (k=3): {test_acc_knn:.4f}")

    mejor_modelo = "Regresión Logística" if test_acc_lr > test_acc_knn else "KNN"
    print(f"\nMejor modelo: {mejor_modelo}")


if __name__ == "__main__":
    main()
