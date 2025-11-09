#!/usr/bin/env python3

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, GBTClassifier, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import time
import json
import sys
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = "data/raw/diabetes_binary_health_indicators_BRFSS2015.csv"
PROCESSED_TRAIN_PATH = "data/processed/train"
PROCESSED_VAL_PATH = "data/processed/val"
PROCESSED_TEST_PATH = "data/processed/test"
RESULTS_PATH = "results/metrics"

# Create directories if they don't exist
os.makedirs("data/processed", exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

# Parse command line arguments
master_url = sys.argv[1] if len(sys.argv) > 1 else None
num_vms = int(sys.argv[2]) if len(sys.argv) > 2 else 1

print("\n" + "="*80)
print("DIABETES PREDICTION - COMPLETE PIPELINE")
print("="*80)
print(f"Configuration: {num_vms} VM(s)")
print("="*80 + "\n")

# ============================================================================
# 1. DATA PREPROCESSING
# ============================================================================

def preprocess_data():
    """Load and preprocess the diabetes dataset"""
    
    print("\n" + "="*80)
    print("STEP 1: DATA PREPROCESSING")
    print("="*80 + "\n")
    
    # Initialize Spark
    print("Initializing Spark...")
    spark = SparkSession.builder \
        .appName("DiabetesPreprocessing") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    # Load data
    print("Loading data...")
    df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)
    
    print(f"Total records: {df.count():,}")
    print(f"Features: {len(df.columns)}")
    df.show(5)
    
    print("\nClass distribution:")
    df.groupBy("Diabetes_binary").count().show()
    
    # Feature engineering
    print("\nPreprocessing features...")
    features = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 
                'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
                'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 
                'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
    
    assembler = VectorAssembler(inputCols=features, outputCol="features_raw", handleInvalid="skip")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
    pipeline = Pipeline(stages=[assembler, scaler])
    
    start = time.time()
    df_final = pipeline.fit(df).transform(df).select("features", col("Diabetes_binary").alias("label"))
    print(f"Transformation done in {time.time()-start:.2f}s")
    
    # Split data
    print("\nSplitting data...")
    train, temp = df_final.randomSplit([0.7, 0.3], seed=42)
    val, test = temp.randomSplit([0.5, 0.5], seed=42)
    
    print(f"Train: {train.count():,}")
    print(f"Val: {val.count():,}")
    print(f"Test: {test.count():,}")
    
    # Save processed data
    print("\nSaving processed data...")
    train.write.mode("overwrite").parquet(PROCESSED_TRAIN_PATH)
    val.write.mode("overwrite").parquet(PROCESSED_VAL_PATH)
    test.write.mode("overwrite").parquet(PROCESSED_TEST_PATH)
    
    print("✓ Preprocessing complete!\n")
    spark.stop()

# ============================================================================
# 2. RANDOM FOREST CLASSIFIER
# ============================================================================

def train_random_forest(master_url, num_vms):
    """Train and evaluate Random Forest classifier"""
    
    print("\n" + "="*80)
    print(f"STEP 2: RANDOM FOREST CLASSIFIER ({num_vms} VMs)")
    print("="*80 + "\n")
    
    # Initialize Spark
    builder = SparkSession.builder.appName(f"DiabetesRF_{num_vms}VM")
    if master_url:
        builder = builder.master(master_url).config("spark.executor.instances", str(num_vms)).config("spark.executor.memory", "2g")
    else:
        builder = builder.master("local[*]")
    spark = builder.config("spark.driver.memory", "4g").getOrCreate()
    
    # Load data
    print("Loading data...")
    train = spark.read.parquet(PROCESSED_TRAIN_PATH).cache()
    test = spark.read.parquet(PROCESSED_TEST_PATH).cache()
    print(f"Train: {train.count():,} | Test: {test.count():,}")
    
    # Train model
    print("\nTraining Random Forest...")
    start = time.time()
    rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100, maxDepth=5, seed=42)
    model = rf.fit(train)
    train_time = time.time() - start
    print(f"Training completed in {train_time:.2f}s")
    
    # Evaluate
    print("\nEvaluating on test set...")
    start = time.time()
    pred = model.transform(test)
    pred_time = time.time() - start
    
    binary_eval = BinaryClassificationEvaluator(labelCol="label")
    multi_eval = MulticlassClassificationEvaluator(labelCol="label")
    
    metrics = {
        "algorithm": "Random Forest",
        "num_vms": num_vms,
        "accuracy": multi_eval.evaluate(pred, {multi_eval.metricName: "accuracy"}),
        "auc_roc": binary_eval.evaluate(pred, {binary_eval.metricName: "areaUnderROC"}),
        "precision": multi_eval.evaluate(pred, {multi_eval.metricName: "weightedPrecision"}),
        "recall": multi_eval.evaluate(pred, {multi_eval.metricName: "weightedRecall"}),
        "f1": multi_eval.evaluate(pred, {multi_eval.metricName: "f1"}),
        "train_time": train_time,
        "pred_time": pred_time
    }
    
    tp = pred.filter((col("prediction") == 1) & (col("label") == 1)).count()
    tn = pred.filter((col("prediction") == 0) & (col("label") == 0)).count()
    fp = pred.filter((col("prediction") == 1) & (col("label") == 0)).count()
    fn = pred.filter((col("prediction") == 0) & (col("label") == 1)).count()
    
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics["confusion"] = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
    
    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS - Random Forest ({num_vms} VMs)")
    print(f"{'='*80}")
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"AUC-ROC:       {metrics['auc_roc']:.4f}")
    print(f"Precision:     {metrics['precision']:.4f}")
    print(f"Recall:        {metrics['recall']:.4f}")
    print(f"F1-Score:      {metrics['f1']:.4f}")
    print(f"Sensitivity:   {metrics['sensitivity']:.4f}")
    print(f"Specificity:   {metrics['specificity']:.4f}")
    print(f"\nTrain Time:    {train_time:.2f}s")
    print(f"Predict Time:  {pred_time:.2f}s")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp:>6,}  |  FN: {fn:>6,}")
    print(f"  FP: {fp:>6,}  |  TN: {tn:>6,}")
    print(f"{'='*80}\n")
    
    # Save results
    with open(f"{RESULTS_PATH}/rf_{num_vms}vm.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Results saved to: {RESULTS_PATH}/rf_{num_vms}vm.json\n")
    spark.stop()

# ============================================================================
# 3. LOGISTIC REGRESSION
# ============================================================================

def train_logistic_regression(master_url, num_vms):
    """Train and evaluate Logistic Regression classifier"""
    
    print("\n" + "="*80)
    print(f"STEP 3: LOGISTIC REGRESSION ({num_vms} VMs)")
    print("="*80 + "\n")
    
    # Initialize Spark
    builder = SparkSession.builder.appName(f"DiabetesLR_{num_vms}VM")
    if master_url:
        builder = builder.master(master_url).config("spark.executor.instances", str(num_vms)).config("spark.executor.memory", "2g")
    else:
        builder = builder.master("local[*]")
    spark = builder.config("spark.driver.memory", "4g").getOrCreate()
    
    # Load data
    print("Loading data...")
    train = spark.read.parquet(PROCESSED_TRAIN_PATH).cache()
    test = spark.read.parquet(PROCESSED_TEST_PATH).cache()
    print(f"Train: {train.count():,} | Test: {test.count():,}")
    
    # Train model
    print("\nTraining Logistic Regression...")
    start = time.time()
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100, regParam=0.0, elasticNetParam=0.0)
    model = lr.fit(train)
    train_time = time.time() - start
    print(f"Training completed in {train_time:.2f}s")
    
    # Evaluate
    print("\nEvaluating on test set...")
    start = time.time()
    pred = model.transform(test)
    pred_time = time.time() - start
    
    binary_eval = BinaryClassificationEvaluator(labelCol="label")
    multi_eval = MulticlassClassificationEvaluator(labelCol="label")
    
    metrics = {
        "algorithm": "Logistic Regression",
        "num_vms": num_vms,
        "accuracy": multi_eval.evaluate(pred, {multi_eval.metricName: "accuracy"}),
        "auc_roc": binary_eval.evaluate(pred, {binary_eval.metricName: "areaUnderROC"}),
        "precision": multi_eval.evaluate(pred, {multi_eval.metricName: "weightedPrecision"}),
        "recall": multi_eval.evaluate(pred, {multi_eval.metricName: "weightedRecall"}),
        "f1": multi_eval.evaluate(pred, {multi_eval.metricName: "f1"}),
        "train_time": train_time,
        "pred_time": pred_time
    }
    
    tp = pred.filter((col("prediction") == 1) & (col("label") == 1)).count()
    tn = pred.filter((col("prediction") == 0) & (col("label") == 0)).count()
    fp = pred.filter((col("prediction") == 1) & (col("label") == 0)).count()
    fn = pred.filter((col("prediction") == 0) & (col("label") == 1)).count()
    
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics["confusion"] = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
    
    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS - Logistic Regression ({num_vms} VMs)")
    print(f"{'='*80}")
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"AUC-ROC:       {metrics['auc_roc']:.4f}")
    print(f"Precision:     {metrics['precision']:.4f}")
    print(f"Recall:        {metrics['recall']:.4f}")
    print(f"F1-Score:      {metrics['f1']:.4f}")
    print(f"Sensitivity:   {metrics['sensitivity']:.4f}")
    print(f"Specificity:   {metrics['specificity']:.4f}")
    print(f"\nTrain Time:    {train_time:.2f}s")
    print(f"Predict Time:  {pred_time:.2f}s")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp:>6,}  |  FN: {fn:>6,}")
    print(f"  FP: {fp:>6,}  |  TN: {tn:>6,}")
    print(f"{'='*80}\n")
    
    # Save results
    with open(f"{RESULTS_PATH}/lr_{num_vms}vm.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Results saved to: {RESULTS_PATH}/lr_{num_vms}vm.json\n")
    spark.stop()

# ============================================================================
# 4. GRADIENT BOOSTING TREES
# ============================================================================

def train_gradient_boosting(master_url, num_vms):
    """Train and evaluate Gradient Boosting Trees classifier"""
    
    print("\n" + "="*80)
    print(f"STEP 4: GRADIENT BOOSTING TREES ({num_vms} VMs)")
    print("="*80 + "\n")
    
    # Initialize Spark
    builder = SparkSession.builder.appName(f"DiabetesGBT_{num_vms}VM")
    if master_url:
        builder = builder.master(master_url).config("spark.executor.instances", str(num_vms)).config("spark.executor.memory", "2g")
    else:
        builder = builder.master("local[*]")
    spark = builder.config("spark.driver.memory", "4g").getOrCreate()
    
    # Load data
    print("Loading data...")
    train = spark.read.parquet(PROCESSED_TRAIN_PATH).cache()
    test = spark.read.parquet(PROCESSED_TEST_PATH).cache()
    print(f"Train: {train.count():,} | Test: {test.count():,}")
    
    # Train model
    print("\nTraining Gradient Boosting Trees...")
    start = time.time()
    gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=20, maxDepth=4, seed=42)
    model = gbt.fit(train)
    train_time = time.time() - start
    print(f"Training completed in {train_time:.2f}s")
    
    # Evaluate
    print("\nEvaluating on test set...")
    start = time.time()
    pred = model.transform(test)
    pred_time = time.time() - start
    
    binary_eval = BinaryClassificationEvaluator(labelCol="label")
    multi_eval = MulticlassClassificationEvaluator(labelCol="label")
    
    metrics = {
        "algorithm": "Gradient Boosting Trees",
        "num_vms": num_vms,
        "accuracy": multi_eval.evaluate(pred, {multi_eval.metricName: "accuracy"}),
        "auc_roc": binary_eval.evaluate(pred, {binary_eval.metricName: "areaUnderROC"}),
        "precision": multi_eval.evaluate(pred, {multi_eval.metricName: "weightedPrecision"}),
        "recall": multi_eval.evaluate(pred, {multi_eval.metricName: "weightedRecall"}),
        "f1": multi_eval.evaluate(pred, {multi_eval.metricName: "f1"}),
        "train_time": train_time,
        "pred_time": pred_time
    }
    
    tp = pred.filter((col("prediction") == 1) & (col("label") == 1)).count()
    tn = pred.filter((col("prediction") == 0) & (col("label") == 0)).count()
    fp = pred.filter((col("prediction") == 1) & (col("label") == 0)).count()
    fn = pred.filter((col("prediction") == 0) & (col("label") == 1)).count()
    
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics["confusion"] = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
    
    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS - Gradient Boosting Trees ({num_vms} VMs)")
    print(f"{'='*80}")
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"AUC-ROC:       {metrics['auc_roc']:.4f}")
    print(f"Precision:     {metrics['precision']:.4f}")
    print(f"Recall:        {metrics['recall']:.4f}")
    print(f"F1-Score:      {metrics['f1']:.4f}")
    print(f"Sensitivity:   {metrics['sensitivity']:.4f}")
    print(f"Specificity:   {metrics['specificity']:.4f}")
    print(f"\nTrain Time:    {train_time:.2f}s")
    print(f"Predict Time:  {pred_time:.2f}s")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp:>6,}  |  FN: {fn:>6,}")
    print(f"  FP: {fp:>6,}  |  TN: {tn:>6,}")
    print(f"{'='*80}\n")
    
    # Save results
    with open(f"{RESULTS_PATH}/gbt_{num_vms}vm.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Results saved to: {RESULTS_PATH}/gbt_{num_vms}vm.json\n")
    spark.stop()

# ============================================================================
# 5. MULTI-LAYER PERCEPTRON CLASSIFIER
# ============================================================================

def train_mlp(master_url, num_vms):
    """Train and evaluate Multi-Layer Perceptron classifier"""
    
    print("\n" + "="*80)
    print(f"STEP 5: MULTI-LAYER PERCEPTRON ({num_vms} VMs)")
    print("="*80 + "\n")
    
    # Initialize Spark
    builder = SparkSession.builder.appName(f"DiabetesMLP_{num_vms}VM")
    if master_url:
        builder = builder.master(master_url).config("spark.executor.instances", str(num_vms)).config("spark.executor.memory", "2g")
    else:
        builder = builder.master("local[*]")
    spark = builder.config("spark.driver.memory", "4g").getOrCreate()
    
    # Load data
    print("Loading data...")
    train = spark.read.parquet(PROCESSED_TRAIN_PATH).cache()
    test = spark.read.parquet(PROCESSED_TEST_PATH).cache()
    print(f"Train: {train.count():,} | Test: {test.count():,}")
    
    # Train model
    print("\nTraining Multi-Layer Perceptron (DNN)...")
    print("Network Architecture: 21 → 32 → 16 → 2")
    start = time.time()
    
    layers = [21, 32, 16, 2]
    mlp = MultilayerPerceptronClassifier(
        featuresCol="features",
        labelCol="label",
        layers=layers,
        maxIter=20,
        blockSize=128,
        seed=42
    )
    
    model = mlp.fit(train)
    train_time = time.time() - start
    print(f"Training completed in {train_time:.2f}s")
    
    # Evaluate
    print("\nEvaluating on test set...")
    start = time.time()
    pred = model.transform(test)
    pred_time = time.time() - start
    
    binary_eval = BinaryClassificationEvaluator(labelCol="label")
    multi_eval = MulticlassClassificationEvaluator(labelCol="label")
    
    metrics = {
        "algorithm": "Multi-Layer Perceptron (DNN)",
        "num_vms": num_vms,
        "network_architecture": "21->32->16->2",
        "accuracy": multi_eval.evaluate(pred, {multi_eval.metricName: "accuracy"}),
        "auc_roc": binary_eval.evaluate(pred, {binary_eval.metricName: "areaUnderROC"}),
        "precision": multi_eval.evaluate(pred, {multi_eval.metricName: "weightedPrecision"}),
        "recall": multi_eval.evaluate(pred, {multi_eval.metricName: "weightedRecall"}),
        "f1": multi_eval.evaluate(pred, {multi_eval.metricName: "f1"}),
        "train_time": train_time,
        "pred_time": pred_time
    }
    
    tp = pred.filter((col("prediction") == 1) & (col("label") == 1)).count()
    tn = pred.filter((col("prediction") == 0) & (col("label") == 0)).count()
    fp = pred.filter((col("prediction") == 1) & (col("label") == 0)).count()
    fn = pred.filter((col("prediction") == 0) & (col("label") == 1)).count()
    
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics["confusion"] = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
    
    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS - Multi-Layer Perceptron ({num_vms} VMs)")
    print(f"{'='*80}")
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"AUC-ROC:       {metrics['auc_roc']:.4f}")
    print(f"Precision:     {metrics['precision']:.4f}")
    print(f"Recall:        {metrics['recall']:.4f}")
    print(f"F1-Score:      {metrics['f1']:.4f}")
    print(f"Sensitivity:   {metrics['sensitivity']:.4f}")
    print(f"Specificity:   {metrics['specificity']:.4f}")
    print(f"\nTrain Time:    {train_time:.2f}s")
    print(f"Predict Time:  {pred_time:.2f}s")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp:>6,}  |  FN: {fn:>6,}")
    print(f"  FP: {fp:>6,}  |  TN: {tn:>6,}")
    print(f"{'='*80}\n")
    
    # Save results
    with open(f"{RESULTS_PATH}/mlp_{num_vms}vm.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Results saved to: {RESULTS_PATH}/mlp_{num_vms}vm.json\n")
    spark.stop()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    start_time = time.time()
    
    # Step 1: Preprocess data (run once)
    if not os.path.exists(PROCESSED_TRAIN_PATH):
        preprocess_data()
    else:
        print("\n✓ Preprocessed data already exists. Skipping preprocessing...\n")
    
    # Step 2-5: Train all algorithms
    train_random_forest(master_url, num_vms)
    train_logistic_regression(master_url, num_vms)
    train_gradient_boosting(master_url, num_vms)
    train_mlp(master_url, num_vms)
    
    # Summary
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nConfiguration: {num_vms} VM(s)")
    print(f"Total Execution Time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"\nResults saved in: {RESULTS_PATH}/")
    print("  • rf_{num_vms}vm.json")
    print("  • lr_{num_vms}vm.json")
    print("  • gbt_{num_vms}vm.json")
    print("  • mlp_{num_vms}vm.json")
    print("\n" + "="*80 + "\n")
