from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import os

# Function: Classification and Save Model by n-gram
def classification(df, feature_cols, label_col, n=None):
    
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_with_features = assembler.transform(df)
    
    train_data, test_data = df_with_features.randomSplit([0.8, 0.2], seed=42)
    
    lr = LogisticRegression(featuresCol="features", labelCol=label_col)
    model = lr.fit(train_data)
    
    predictions = model.transform(test_data)
    
    evaluator = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="accuracy"
    )
    
    accuracy = evaluator.evaluate(predictions)
    
    print(f"Test Accuracy = {accuracy:.4f}")

    # Save the model if n is provided
    if n is not None:
        model_folder = f"models/{n}-gram_model"
        if not os.path.exists(model_folder):
            model.save(model_folder)
            print(f"Model saved to {model_folder}")
        else:
            print(f"⚠️ Save path {model_folder} already exists. Choose a different folder or delete the existing one.")

    return model, accuracy
