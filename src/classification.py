import os
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def classification(df, feature_cols, label_col, n=None):
    # ✅ Ukloni nevalidne kolone
    input_features = [f for f in feature_cols if isinstance(f, str) and f.strip() != ""]

    if not input_features:
        raise ValueError("❌ No valid input features provided for classification.")

    # 🔧 Vektorizacija
    assembler = VectorAssembler(inputCols=input_features, outputCol="features")
    df_with_features = assembler.transform(df)

    # 🔁 StringIndexer za labelu
    indexer = StringIndexer(inputCol=label_col, outputCol="label_index")
    df_indexed = indexer.fit(df_with_features).transform(df_with_features)

    # 🔀 Podela na train/test
    train_data, test_data = df_indexed.randomSplit([0.8, 0.2], seed=42)

    # 🧠 Model
    lr = LogisticRegression(featuresCol="features", labelCol="label_index")
    model = lr.fit(train_data)

    # 📊 Evaluacija
    predictions = model.transform(test_data)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label_index", predictionCol="prediction", metricName="accuracy"
    )
    accuracy = evaluator.evaluate(predictions)
    print(f"✅ Test Accuracy = {accuracy:.4f}")

    # 💾 Čuvanje modela
    if n is not None:
        model_folder = f"models/{n}-gram_model"
        os.makedirs("models", exist_ok=True)
        if not os.path.exists(model_folder):
            model.save(model_folder)
            print(f"✅ Model saved to {model_folder}")
        else:
            print(f"⚠️ Save path {model_folder} already exists. Choose a different folder or delete the existing one.")

    return model, accuracy
