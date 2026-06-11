from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import os


def main():
    spark = SparkSession.builder \
        .appName("MultiPlatformSentimentAnalysis") \
        .getOrCreate()

    # CSV already in your data/ folder
    data_path = "data/multi_platform_social_sentiment_evolution.csv"

    # 1. Load CSV
    df_raw = spark.read.csv(
        data_path,
        header=True,
        inferSchema=True
    )

    print("Raw schema:")
    df_raw.printSchema()
    df_raw.show(5, truncate=80)

    # 2. Select useful columns (NO sentiment_positive/negative/neutral here)
    df = df_raw.select(
        "platform",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "followers",
        "account_age_days",
        "verified",
        "topic",
        "language",
        "content_length",
        "media_type",
        "num_hashtags",
        "sentiment_category",  # label only
        "likes",
        "shares",
        "comments",
        "views",
        "total_engagement",
        "engagement_rate_per_1k_followers",
        "hours_since_post",
        "viral_coefficient",
        "cross_platform_spread",
        "toxicity_score"
    )

    # Drop rows with missing label or platform
    df = df.na.drop(subset=["platform", "sentiment_category"])

    # Optional: sample to keep training manageable
    df = df.orderBy(rand()).limit(200000)

    print("After cleaning & sampling, total rows:", df.count())

    # 3. Cast numeric columns
    numeric_cols = [
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "followers",
        "account_age_days",
        "content_length",
        "num_hashtags",
        "likes",
        "shares",
        "comments",
        "views",
        "total_engagement",
        "engagement_rate_per_1k_followers",
        "hours_since_post",
        "viral_coefficient",
        "cross_platform_spread",
        "toxicity_score"
    ]

    for c in numeric_cols:
        df = df.withColumn(c, col(c).cast("double"))

    # Fill any remaining null numeric values with 0
    df = df.fillna({c: 0.0 for c in numeric_cols})

    # 4. Categorical columns
    cat_cols = ["platform", "topic", "language", "media_type", "verified"]

    # Index categorical features
    indexers = [
        StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
        for c in cat_cols
    ]

    # Index label (negative / neutral / positive)
    label_indexer = StringIndexer(
        inputCol="sentiment_category",
        outputCol="label_index",
        handleInvalid="keep"
    )

    # One-hot encode categorical indices
    encoder = OneHotEncoder(
        inputCols=[c + "_idx" for c in cat_cols],
        outputCols=[c + "_ohe" for c in cat_cols]
    )

    # Assemble feature vector
    feature_cols = numeric_cols + [c + "_ohe" for c in cat_cols]

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )

    # 5. Multiclass Logistic Regression
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label_index",
        maxIter=30
    )

    # 6. Build pipeline
    stages = []
    stages.extend(indexers)
    stages.append(label_indexer)
    stages.append(encoder)
    stages.append(assembler)
    stages.append(lr)

    pipeline = Pipeline(stages=stages)

    # 7. Train/test split
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    print("Train:", train_df.count())
    print("Test :", test_df.count())

    # 8. Train model
    model = pipeline.fit(train_df)

    # 9. Predictions
    predictions = model.transform(test_df)
    predictions.select("platform", "sentiment_category", "prediction") \
               .show(10, truncate=80)

    # 10. Evaluation: accuracy, precision, recall, F1 (weighted)
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="label_index",
        predictionCol="prediction",
        metricName="accuracy"
    )
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="label_index",
        predictionCol="prediction",
        metricName="f1"
    )
    evaluator_prec = MulticlassClassificationEvaluator(
        labelCol="label_index",
        predictionCol="prediction",
        metricName="weightedPrecision"
    )
    evaluator_rec = MulticlassClassificationEvaluator(
        labelCol="label_index",
        predictionCol="prediction",
        metricName="weightedRecall"
    )

    accuracy = evaluator_acc.evaluate(predictions)
    precision = evaluator_prec.evaluate(predictions)
    recall = evaluator_rec.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)

    print("\n=== Metrics ===")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    # 11. Save results
    os.makedirs("results", exist_ok=True)

    # Metrics
    with open("results/metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy : {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall   : {recall:.4f}\n")
        f.write(f"F1 Score : {f1:.4f}\n")

    # Overall sentiment distribution
    label_counts = df.groupBy("sentiment_category").count()
    label_counts.toPandas().to_csv("results/label_distribution.csv", index=False)

    # Per-platform sentiment distribution
    platform_dist = df.groupBy("platform", "sentiment_category").count()
    platform_dist.toPandas().to_csv("results/platform_sentiment.csv", index=False)

    # Sample predictions
    predictions.select("platform", "sentiment_category", "prediction") \
        .limit(1000) \
        .toPandas().to_csv(
            "results/predictions_sample.csv",
            index=False,
            encoding="utf-8"
        )

    print("\nFiles saved in results/")

    spark.stop()


if __name__ == "__main__":
    main()
