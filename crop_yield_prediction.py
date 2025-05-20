from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, when, count
from pyspark.ml.feature import (VectorAssembler, StandardScaler, 
                               StringIndexer, OneHotEncoder)
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import logging
import tempfile
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_spark_session():
    """Create optimized Spark session with MongoDB configuration"""
    temp_dir = tempfile.mkdtemp()
    
    return SparkSession.builder \
        .appName("EnhancedCropYieldPrediction") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.local.dir", temp_dir) \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.1.0") \
        .config("spark.cleaner.periodicGC.interval", "1min") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

def load_data_from_mongodb(spark):
    """Load and validate data from MongoDB"""
    try:
        df = spark.read.format("mongodb") \
            .option("database", "agriculture_db") \
            .option("collection", "crop_data") \
            .option("uri", "mongodb://127.0.0.1:27017") \
            .load()
        
        logger.info(f"Successfully loaded {df.count()} records from MongoDB")
        return df
    except Exception as e:
        logger.error(f"Error loading data from MongoDB: {str(e)}")
        raise

def explore_data(df):
    """Perform comprehensive data exploration"""
    logger.info("=== Data Exploration ===")
    
    print(f"Total records: {df.count()}")
    print(f"Schema:")
    df.printSchema()
    
    print("\nSample data:")
    df.show(5, truncate=False)
    
    print("\nMissing values check:")
    df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()
    
    numerical_cols = [f.name for f in df.schema.fields if f.dataType != 'string']
    print("\nNumerical columns statistics:")
    df.select(numerical_cols).describe().show()
    
    return df

def clean_and_prepare_data(df, numerical_features, categorical_features, target_col):
    """Clean and prepare data"""
    logger.info("=== Data Preparation ===")
    
    required_cols = numerical_features + categorical_features + [target_col]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    initial_count = df.count()
    df_clean = df.select(*required_cols).dropna()
    logger.info(f"Removed {initial_count - df_clean.count()} rows with null values")
    
    return df_clean.cache()

def build_feature_pipeline(numerical_features, categorical_features):
    """Build feature processing pipeline"""
    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep")
        for col in categorical_features
    ]
    
    encoders = [
        OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_encoded")
        for col in categorical_features
    ]
    
    assembler = VectorAssembler(
        inputCols=numerical_features + [f"{col}_encoded" for col in categorical_features],
        outputCol="features",
        handleInvalid="skip"
    )
    
    return Pipeline(stages=indexers + encoders + [assembler])

def build_model(feature_pipeline, target_col):
    """Build the complete model pipeline"""
    # Base models
    lr = LinearRegression(
        labelCol=target_col,
        featuresCol="features",
        predictionCol="prediction"
    )
    
    rf = RandomForestRegressor(
        labelCol=target_col,
        featuresCol="features",
        predictionCol="prediction"
    )
    
    # Complete pipeline
    model_pipeline = Pipeline(
        stages=feature_pipeline.getStages() + [lr]
    )
    
    return model_pipeline

def build_cross_validator(model_pipeline, target_col):
    """Build cross validator for model tuning"""
    # Get the LinearRegression estimator from the pipeline
    lr_estimator = model_pipeline.getStages()[-1]
    
    # Build parameter grid using the correct syntax
    param_grid = (ParamGridBuilder()
                 .addGrid(lr_estimator.regParam, [0.01, 0.1])
                 .addGrid(lr_estimator.elasticNetParam, [0.0, 0.5])
                 .build())
    
    evaluator = RegressionEvaluator(
        labelCol=target_col,
        predictionCol="prediction",
        metricName="rmse"
    )
    
    return CrossValidator(
        estimator=model_pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=4,
        seed=42
    )

def train_and_evaluate_model(df, numerical_features, categorical_features, target_col):
    """Train and evaluate model"""
    logger.info("=== Model Training ===")
    
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
    train_data.cache()
    test_data.cache()
    
    logger.info(f"Training set size: {train_data.count()}")
    logger.info(f"Test set size: {test_data.count()}")
    
    # Build pipelines
    feature_pipeline = build_feature_pipeline(numerical_features, categorical_features)
    model_pipeline = build_model(feature_pipeline, target_col)
    cv = build_cross_validator(model_pipeline, target_col)
    
    # Train model
    model = cv.fit(train_data)
    
    # Make predictions
    train_predictions = model.transform(train_data)
    test_predictions = model.transform(test_data)
    
    # Evaluate
    evaluator = RegressionEvaluator(
        labelCol=target_col,
        predictionCol="prediction",
        metricName="rmse"
    )
    
    train_rmse = evaluator.evaluate(train_predictions)
    test_rmse = evaluator.evaluate(test_predictions)
    
    print("\n=== Model Performance ===")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    
    # Show sample predictions
    print("\nSample predictions:")
    test_predictions.select(target_col, "prediction").show(100)
    
    return model, test_predictions

def main():
    """Main execution function"""
    NUMERICAL_FEATURES = ["Production", "Area", "Temperature"]
    CATEGORICAL_FEATURES = ["Country", "Crop"]
    TARGET_COL = "Yield"
    
    spark = None
    try:
        spark = create_spark_session()
        
        df = load_data_from_mongodb(spark)
        df = explore_data(df)
        df_clean = clean_and_prepare_data(df, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COL)
        
        model, predictions = train_and_evaluate_model(
            df_clean, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COL
        )
        
        # Create directory if it doesn't exist
        import os
        os.makedirs("saved", exist_ok=True)
        
        model.save("saved/optimized_model")
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
    finally:
        if spark:
            spark.stop()
            logger.info("Spark session stopped")

if __name__ == "__main__":
    main()
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, isnan, when, count, percentile_approx, stddev, mean
# from pyspark.sql.types import DoubleType
# from pyspark.ml.feature import (VectorAssembler, StandardScaler, 
#                                StringIndexer, OneHotEncoder)
# from pyspark.ml.regression import LinearRegression, RandomForestRegressor
# from pyspark.ml.evaluation import RegressionEvaluator
# from pyspark.ml import Pipeline
# from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
# import logging
# import tempfile
# import shutil
# import os

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def create_spark_session():
#     """Create optimized Spark session with MongoDB configuration"""
#     temp_dir = tempfile.mkdtemp()
    
#     return SparkSession.builder \
#         .appName("EnhancedCropYieldPrediction") \
#         .config("spark.sql.adaptive.enabled", "true") \
#         .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
#         .config("spark.sql.adaptive.skewJoin.enabled", "true") \
#         .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
#         .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
#         .config("spark.local.dir", temp_dir) \
#         .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.1.0") \
#         .config("spark.cleaner.periodicGC.interval", "1min") \
#         .config("spark.driver.memory", "4g") \
#         .config("spark.executor.memory", "4g") \
#         .getOrCreate()

# def load_data_from_mongodb(spark):
#     """Load and validate data from MongoDB"""
#     try:
#         df = spark.read.format("mongodb") \
#             .option("database", "agriculture_db") \
#             .option("collection", "crop_data") \
#             .option("uri", "mongodb://127.0.0.1:27017") \
#             .load()
        
#         logger.info(f"Successfully loaded {df.count()} records from MongoDB")
#         return df
#     except Exception as e:
#         logger.error(f"Error loading data from MongoDB: {str(e)}")
#         raise

# def explore_data(df):
#     """Perform comprehensive data exploration"""
#     logger.info("=== Data Exploration ===")
    
#     print(f"Total records: {df.count()}")
#     print(f"Schema:")
#     df.printSchema()
    
#     print("\nSample data:")
#     df.show(5, truncate=False)
    
#     print("\nMissing values check:")
#     df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()
    
#     # Get numerical columns (excluding string types)
#     numerical_cols = []
#     for field in df.schema.fields:
#         if field.dataType.typeName() in ['double', 'float', 'integer', 'long']:
#             numerical_cols.append(field.name)
    
#     if numerical_cols:
#         print("\nNumerical columns statistics:")
#         df.select(numerical_cols).describe().show()
    
#     return df

# def remove_outliers(df, target_col, numerical_features):
#     """Remove outliers using IQR method"""
#     logger.info("=== Removing Outliers ===")
    
#     initial_count = df.count()
    
#     # Remove outliers for target variable
#     stats = df.select(
#         percentile_approx(target_col, 0.25).alias("q1"),
#         percentile_approx(target_col, 0.75).alias("q3"),
#         mean(target_col).alias("mean"),
#         stddev(target_col).alias("stddev")
#     ).collect()[0]
    
#     q1, q3 = stats["q1"], stats["q3"]
#     iqr = q3 - q1
#     lower_bound = q1 - 1.5 * iqr
#     upper_bound = q3 + 1.5 * iqr
    
#     logger.info(f"Target column {target_col} - Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
#     logger.info(f"Outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
#     # Filter outliers
#     df_filtered = df.filter(
#         (col(target_col) >= lower_bound) & 
#         (col(target_col) <= upper_bound) &
#         (col(target_col) > 0)  # Remove negative yields
#     )
    
#     # Also remove outliers from numerical features
#     for feature in numerical_features:
#         if feature != target_col:
#             feature_stats = df_filtered.select(
#                 percentile_approx(feature, 0.05).alias("p5"),
#                 percentile_approx(feature, 0.95).alias("p95")
#             ).collect()[0]
            
#             p5, p95 = feature_stats["p5"], feature_stats["p95"]
#             df_filtered = df_filtered.filter(
#                 (col(feature) >= p5) & (col(feature) <= p95) & (col(feature) > 0)
#             )
    
#     final_count = df_filtered.count()
#     logger.info(f"Removed {initial_count - final_count} outlier records ({((initial_count - final_count) / initial_count * 100):.1f}%)")
    
#     return df_filtered

# def clean_and_prepare_data(df, numerical_features, categorical_features, target_col):
#     """Clean and prepare data with outlier removal"""
#     logger.info("=== Data Preparation ===")
    
#     required_cols = numerical_features + categorical_features + [target_col]
#     missing_cols = set(required_cols) - set(df.columns)
#     if missing_cols:
#         raise ValueError(f"Missing required columns: {missing_cols}")
    
#     # Select required columns and remove nulls
#     initial_count = df.count()
#     df_clean = df.select(*required_cols).dropna()
#     logger.info(f"Removed {initial_count - df_clean.count()} rows with null values")
    
#     # Remove outliers
#     df_clean = remove_outliers(df_clean, target_col, numerical_features)
    
#     # Cast numerical columns to double to ensure consistency
#     for col_name in numerical_features + [target_col]:
#         df_clean = df_clean.withColumn(col_name, col(col_name).cast(DoubleType()))
    
#     return df_clean.cache()

# def build_feature_pipeline(numerical_features, categorical_features):
#     """Build feature processing pipeline with scaling"""
#     indexers = [
#         StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep")
#         for col in categorical_features
#     ]
    
#     encoders = [
#         OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_encoded")
#         for col in categorical_features
#     ]
    
#     # Assemble features
#     assembler = VectorAssembler(
#         inputCols=numerical_features + [f"{col}_encoded" for col in categorical_features],
#         outputCol="features_raw",
#         handleInvalid="skip"
#     )
    
#     # Scale features
#     scaler = StandardScaler(
#         inputCol="features_raw",
#         outputCol="features",
#         withStd=True,
#         withMean=True
#     )
    
#     return Pipeline(stages=indexers + encoders + [assembler, scaler])

# def build_model(feature_pipeline, target_col, model_type="linear"):
#     """Build the complete model pipeline"""
#     if model_type == "linear":
#         estimator = LinearRegression(
#             labelCol=target_col,
#             featuresCol="features",
#             predictionCol="prediction",
#             regParam=0.01,  # Add regularization
#             elasticNetParam=0.1
#         )
#     else:
#         estimator = RandomForestRegressor(
#             labelCol=target_col,
#             featuresCol="features",
#             predictionCol="prediction",
#             numTrees=50,
#             maxDepth=10,
#             seed=42
#         )
    
#     # Complete pipeline
#     model_pipeline = Pipeline(
#         stages=feature_pipeline.getStages() + [estimator]
#     )
    
#     return model_pipeline

# def build_cross_validator(model_pipeline, target_col, model_type="linear"):
#     """Build cross validator for model tuning"""
#     estimator = model_pipeline.getStages()[-1]
    
#     if model_type == "linear":
#         param_grid = (ParamGridBuilder()
#                      .addGrid(estimator.regParam, [0.001, 0.01, 0.1])
#                      .addGrid(estimator.elasticNetParam, [0.0, 0.1, 0.5])
#                      .build())
#     else:
#         param_grid = (ParamGridBuilder()
#                      .addGrid(estimator.numTrees, [30, 50, 100])
#                      .addGrid(estimator.maxDepth, [5, 10, 15])
#                      .build())
    
#     evaluator = RegressionEvaluator(
#         labelCol=target_col,
#         predictionCol="prediction",
#         metricName="rmse"
#     )
    
#     return CrossValidator(
#         estimator=model_pipeline,
#         estimatorParamMaps=param_grid,
#         evaluator=evaluator,
#         numFolds=3,
#         parallelism=2,  # Reduced parallelism to avoid memory issues
#         seed=42
#     )

# def evaluate_model_comprehensive(predictions, target_col):
#     """Comprehensive model evaluation"""
#     evaluator_rmse = RegressionEvaluator(
#         labelCol=target_col,
#         predictionCol="prediction",
#         metricName="rmse"
#     )
    
#     evaluator_mae = RegressionEvaluator(
#         labelCol=target_col,
#         predictionCol="prediction",
#         metricName="mae"
#     )
    
#     evaluator_r2 = RegressionEvaluator(
#         labelCol=target_col,
#         predictionCol="prediction",
#         metricName="r2"
#     )
    
#     rmse = evaluator_rmse.evaluate(predictions)
#     mae = evaluator_mae.evaluate(predictions)
#     r2 = evaluator_r2.evaluate(predictions)
    
#     return rmse, mae, r2

# def train_and_evaluate_model(df, numerical_features, categorical_features, target_col):
#     """Train and evaluate model with comprehensive metrics"""
#     logger.info("=== Model Training ===")
    
#     # Split data
#     train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
#     train_data.cache()
#     test_data.cache()
    
#     logger.info(f"Training set size: {train_data.count()}")
#     logger.info(f"Test set size: {test_data.count()}")
    
#     # Try both models
#     results = {}
    
#     for model_type in ["linear", "random_forest"]:
#         logger.info(f"\n=== Training {model_type.upper()} model ===")
        
#         # Build pipelines
#         feature_pipeline = build_feature_pipeline(numerical_features, categorical_features)
#         model_pipeline = build_model(feature_pipeline, target_col, model_type)
#         cv = build_cross_validator(model_pipeline, target_col, model_type)
        
#         # Train model
#         model = cv.fit(train_data)
        
#         # Make predictions
#         train_predictions = model.transform(train_data)
#         test_predictions = model.transform(test_data)
        
#         # Evaluate
#         train_rmse, train_mae, train_r2 = evaluate_model_comprehensive(train_predictions, target_col)
#         test_rmse, test_mae, test_r2 = evaluate_model_comprehensive(test_predictions, target_col)
        
#         results[model_type] = {
#             'model': model,
#             'train_rmse': train_rmse,
#             'test_rmse': test_rmse,
#             'train_mae': train_mae,
#             'test_mae': test_mae,
#             'train_r2': train_r2,
#             'test_r2': test_r2,
#             'predictions': test_predictions
#         }
        
#         print(f"\n=== {model_type.upper()} Model Performance ===")
#         print(f"Training RMSE: {train_rmse:.4f}")
#         print(f"Test RMSE: {test_rmse:.4f}")
#         print(f"Training MAE: {train_mae:.4f}")
#         print(f"Test MAE: {test_mae:.4f}")
#         print(f"Training R²: {train_r2:.4f}")
#         print(f"Test R²: {test_r2:.4f}")
#         print(f"Overfitting ratio (Test RMSE / Train RMSE): {test_rmse / train_rmse:.2f}")
    
#     # Select best model based on test RMSE
#     best_model_type = min(results.keys(), key=lambda k: results[k]['test_rmse'])
#     best_model = results[best_model_type]['model']
#     best_predictions = results[best_model_type]['predictions']
    
#     print(f"\n=== Best Model: {best_model_type.upper()} ===")
#     print("Sample predictions:")
#     best_predictions.select(target_col, "prediction").show(10)
    
#     return best_model, best_predictions

# def main():
#     """Main execution function"""
#     NUMERICAL_FEATURES = ["Production", "Area", "Temperature"]
#     CATEGORICAL_FEATURES = ["Country", "Crop"]
#     TARGET_COL = "Yield"
    
#     spark = None
#     try:
#         spark = create_spark_session()
        
#         df = load_data_from_mongodb(spark)
#         df = explore_data(df)
#         df_clean = clean_and_prepare_data(df, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COL)
        
#         model, predictions = train_and_evaluate_model(
#             df_clean, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COL
#         )
        
#         # Create directory if it doesn't exist
#         os.makedirs("saved", exist_ok=True)
        
#         model.write().overwrite().save("saved/optimized_model")
#         logger.info("Pipeline completed successfully!")
        
#     except Exception as e:
#         logger.error(f"Pipeline failed: {str(e)}")
#         raise
#     finally:
#         if spark:
#             spark.stop()
#             logger.info("Spark session stopped")

# if __name__ == "__main__":
#     main()