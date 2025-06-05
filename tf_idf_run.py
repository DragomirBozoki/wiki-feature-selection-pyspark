from src.preprocessing import spark_load, tokenization
from src.label import label_text
from src.tf_idf import tfidf_classification
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType

if __name__ == "__main__":
    print("🔁 Starting Spark session...")
    spark, df = spark_load()

    print("🔡 Tokenizing text...")
    df_tokens = tokenization(df)

    label_udf = udf(label_text, StringType())
    df_tokens = df_tokens.withColumn("label", label_udf(col("text")))

    print("🏁 Running TF-IDF classification...")
    pipeline, acc = tfidf_classification(df_tokens)
