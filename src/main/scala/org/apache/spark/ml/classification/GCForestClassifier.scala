package org.apache.spark.ml.classification

import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param._
import org.apache.spark.ml.tree.GCForestParams
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable, MLWritable, MLWriter}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{LongType, StructField, StructType}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by chengli on 3/3/17.
  */
class GCForestClassifier(override val uid: String)
  extends ProbabilisticClassifier[Vector, GCForestClassifier, GCForestClassificationModel]
  with DefaultParamsWritable with GCForestParams {

  def this() = this(Identifiable.randomUID("gcf"))

  def extractMatrixRDD(dataset: Dataset[_], windowWidth: Int, windowHeight: Int): DataFrame = {
    require(getDataSize.length == 2, "You must set Image resolution by setDataSize")
    val sparkSession = dataset.sparkSession
    val schema = dataset.schema

    val width = getDataSize(0)
    val height = getDataSize(1)

    require(width >= windowWidth, "The width of image must be greater than window's")
    require(height >= windowHeight, "The heigth of image must be greater than window's")

    val vector2Matrix = udf { (features: Vector, w: Int, h: Int) =>
      new DenseMatrix(w, h, features.toArray)
    }


    val windowInstances = dataset.withColumn($(featuresCol), vector2Matrix(col($(featuresCol))))
      .rdd.zipWithIndex.flatMap {
        case (row, index) =>
          val rows = ArrayBuffer[Row]()
          val featureIdx = row.fieldIndex($(featuresCol))
          val matrix = row.getAs[DenseMatrix]($(featuresCol))

          // TODO should be optimized
          Range(0, width - windowWidth + 1).foreach { x_offset =>
            Range(0, height - windowHeight + 1).foreach { y_offset =>
              val features = Array.fill[Double](windowWidth * windowHeight)(0)
              Range(0, windowWidth).foreach { x =>
                Range(0, windowHeight).foreach { y =>
                  features(x * windowWidth + y * windowHeight) = matrix(x + x_offset, y + y_offset)
                  val newRow = ArrayBuffer[Any]() += row.toSeq
                  newRow(featureIdx) = new DenseVector(features)
                  newRow += (x_offset * (width - windowWidth + 1) + y_offset * (height - windowHeight + 1)).toLong
                  newRow += index
                  rows += Row.fromSeq(newRow)
                }
              }
            }
          }
          rows
    }

    val newSchema = schema
      .add(StructField($(windowCol), LongType))
      .add(StructField($(instanceCol), LongType))

    sparkSession.createDataFrame(windowInstances, newSchema)
  }

  def featureTransform(dataset: Dataset[_], rfc: RandomForestClassifier)
  : (DataFrame, RandomForestClassificationModel) = {
    val schema = dataset.schema
    val sparkSession = dataset.sparkSession
    var out: DataFrame = null

    val splits = MLUtils.kFold(dataset.toDF.rdd, $(numFolds), $(seed))
    splits.zipWithIndex.foreach { case ((training, validation), splitIndex) =>
      val trainingDataset = sparkSession.createDataFrame(training, schema).cache
      val validationDataset = sparkSession.createDataFrame(validation, schema).cache
      val model = rfc.fit(trainingDataset)

      trainingDataset.unpersist()
      val predict = model.transform(validationDataset).drop($(featuresCol))
        .withColumnRenamed($(probabilityCol), $(featuresCol))
      out = if (out == null) predict else out.union(predict)
      validationDataset.unpersist()
    }
    (out, rfc.fit(dataset))
  }

  def genRFClassifier(rfType: String,
                          treeNum: Int,
                          minInstancePerNode: Int
                         ): RandomForestClassifier = {
    val rf = rfType match {
      case "rf" => new RandomForestClassifier()
      case "crtf" => new CompleteRandomTreeForestClassifier()
    }

    rf.setNumTrees(treeNum)
      .setMaxBins(Int.MaxValue)
      .setMaxDepth(30)
      .setMinInstancesPerNode(minInstancePerNode)
      .setFeatureSubsetStrategy("sqrt")
  }

  def concatenate(dataset: Dataset[_], sets: Dataset[_]*): DataFrame = {
    val sparkSession = dataset.sparkSession
    var unionSet = dataset.toDF
    sets.foreach(ds => unionSet = unionSet.union(ds.toDF))

    class Record(val instance: Long,
                 val label: Double,
                 val features: Vector,
                 val scanId: Int,
                 val treeNum: Int,
                 val winId: Long)

    val concatData = unionSet.select(
      $(instanceCol), $(labelCol),
      $(featuresCol), $(scanCol),
      $(treeNumCol), $(windowCol)).rdd.map {
      row =>
        val instance = row.getAs[Long]($(instanceCol))
        val label = row.getAs[Double]($(labelCol))
        val features = row.getAs[Vector]($(featuresCol))
        val scanId = row.getAs[Int]($(scanCol))
        val treeNum = row.getAs[Int]($(treeNumCol))
        val winId = row.getAs[Long]($(windowCol))

        new Record(instance, label, features, scanId, treeNum, winId)
    }.groupBy(
      record => record.instance
    ).map { group =>
      val instance = group._1
      val records = group._2
      val label = records.head.label

      def recordCompare(left: Record, right: Record): Boolean = {
        var code = left.scanId.compareTo(right.scanId)
        if (code == 0) code = left.treeNum.compareTo(right.treeNum)
        if (code == 0) code = left.winId.compareTo(right.winId)
        code < 0
      }

      val features = new DenseVector(records.toSeq.sortWith(recordCompare).flatMap(_.features.toArray).toArray)

      Row.fromSeq(Array[Any](instance, label, features))
    }

    val schema: StructType = StructType(Seq[StructField]())
      .add(StructField($(instanceCol), LongType))
      .add(StructField($(labelCol), LongType))
      .add(StructField($(featuresCol), LongType))
    sparkSession.createDataFrame(concatData, schema)
  }

  def concatPredict(dataset: Dataset[_]): DataFrame = {
    val sparkSession = dataset.sparkSession
    val schema = new StructType()
      .add(StructField($(instanceCol), LongType))
      .add(StructField($(featuresCol), new VectorUDT))

    val predictionRDD = dataset.toDF.rdd.groupBy(_.getAs[Long]($(instanceCol))).map { group =>
      val instanceId = group._1
      val rows = group._2

      val features = new DenseVector(rows.toArray
        .sortWith(_.getAs[Int]($(treeNumCol)) < _.getAs[Int]($(treeNumCol)))
        .flatMap(_.getAs[Vector]($(featuresCol)).toArray))
      Row.fromSeq(Array[Any](instanceId, features))
    }
    sparkSession.createDataFrame(predictionRDD, schema)
  }

  def mergeFeatureAndPredict(feature: Dataset[_], predict: Dataset[_]): DataFrame = {
    val vectorMerge = udf { (v1: Vector, v2: Vector) =>
      new DenseVector(v1.toArray.union(v2.toArray))
    }

    feature.join(
      predict.withColumnRenamed($(featuresCol), $(predictionCol)),
      Seq($(instanceCol))
    ).withColumn(
      $(featuresCol), vectorMerge(col($(featuresCol)), col($(predictionCol)))
    ).select($(instanceCol), $(featuresCol), $(labelCol))
  }

  override protected def train(dataset: Dataset[_]): GCForestClassificationModel = {
    val numClasses: Int = getNumClasses(dataset)
    var scanFeature: Dataset[_] = null
    val mgsModels = ArrayBuffer[MultiGrainedScanModel]()
    val erfModels = ArrayBuffer[Array[RandomForestClassificationModel]]()

    /**
      *  Multi-Grained Scanning
      */
    if ($(dataStyle) == "image") {
      require($(multiScanWindow).length % 2 == 0) // TODO comment

      val scanFeatures = ArrayBuffer[Dataset[_]]()

      Range(0, $(multiScanWindow).length / 2).foreach { i =>
        val (w, h) = (i, i+1)
        val windowInstances = extractMatrixRDD(dataset, w, h)

        val rf = genRFClassifier("rf", $(scanForestTreeNum), $(scanForestMinInstancesPerNode))
        var (rfFeature, rfModel) = featureTransform(windowInstances, rf)
        rfFeature = rfFeature.withColumn($(treeNumCol), lit(1)).withColumn($(scanCol), lit(i))
        scanFeatures += rfFeature

        val crtf = genRFClassifier("crtf", $(cascadeForestTreeNum), $(cascadeForestMinInstancesPerNode))
        var (crtfFeature, crtfModel) = featureTransform(windowInstances, crtf)
        crtfFeature = crtfFeature.withColumn($(treeNumCol), lit(2)).withColumn($(scanCol), lit(i))
        scanFeatures += crtfFeature

        mgsModels += new MultiGrainedScanModel(w, h, rfModel, crtfModel)
      }

      scanFeature = concatenate(scanFeatures.head, scanFeatures.tail:_*).cache

    } else if ($(dataStyle) == "sequence") { // TODO
      throw new UnsupportedOperationException("Unsupported sequence data rightly!")
    } else {
      throw new UnsupportedOperationException("The dataStyle : " + $(dataStyle) + " is unsupported!")
    }

    /**
      *  Cascade Forest
     */

    val sparkSession = scanFeature.sparkSession
    var lastPrediction: DataFrame = null

    Range(0, $(cascadeForestMaxIteration)).foreach { it =>
      val ensembleRandomForest = Array[RandomForestClassifier](
        genRFClassifier("rf", $(cascadeForestTreeNum), $(cascadeForestMinInstancesPerNode)),
        genRFClassifier("rf", $(cascadeForestTreeNum), $(cascadeForestMinInstancesPerNode)),
        genRFClassifier("crtf", $(cascadeForestTreeNum), $(cascadeForestMinInstancesPerNode)),
        genRFClassifier("crtf", $(cascadeForestTreeNum), $(cascadeForestMinInstancesPerNode))
      )

      val training = mergeFeatureAndPredict(scanFeature, lastPrediction)
      var Array(growingSet, estimatingSet) = training.randomSplit(Array(0.8d, 0.2d), $(seed))

      val models = ensembleRandomForest.map(classifier => classifier.fit(growingSet))
      erfModels += models

      var ensemblePredict: DataFrame = null
      ensembleRandomForest.indices.foreach { it =>
        val predict = featureTransform(training, ensembleRandomForest(it))._1
          .withColumn($(treeNumCol), lit(it))
          .select($(instanceCol), $(featuresCol), $(treeNumCol))

        ensemblePredict = if (ensemblePredict == null) predict else ensemblePredict.toDF.union(predict)
      }

      val schema = new StructType()
        .add(StructField($(instanceCol), LongType))
        .add(StructField($(featuresCol), new VectorUDT))

      val predictionRDD = ensemblePredict.rdd.groupBy(_.getAs[Long]($(instanceCol))).map { group =>
        val instanceId = group._1
        val rows = group._2

        val features = new DenseVector(rows.toArray
          .sortWith(_.getAs[Int]($(treeNumCol)) < _.getAs[Int]($(treeNumCol)))
          .flatMap(_.getAs[Vector]($(featuresCol)).toArray))
        Row.fromSeq(Array[Any](instanceId, features))
      }
      lastPrediction = sparkSession.createDataFrame(predictionRDD, schema)
    }

    scanFeature.unpersist
    new GCForestClassificationModel(mgsModels.toArray, erfModels.toArray, numClasses)
  }

  override def copy(extra: ParamMap): GCForestClassifier = defaultCopy(extra)
}


class GCForestClassificationModel private[ml] ( override val uid: String,
                                                private val scanModel: Array[MultiGrainedScanModel],
                                                private val cascadeForest: Array[Array[RandomForestClassificationModel]],
                                                override val numClasses: Int)
  extends ProbabilisticClassificationModel[Vector, GCForestClassificationModel]
    with GCForestParams /*with MLWritable */with Serializable {

  def this(
    scanModel: Array[MultiGrainedScanModel],
    cascadeForest: Array[Array[RandomForestClassificationModel]],
    numClasses: Int) =
    this(Identifiable.randomUID("gcfc"), scanModel, cascadeForest, numClasses)

  override def predictRaw(features: Vector): Vector = features

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        ProbabilisticClassificationModel.normalizeToProbabilitiesInPlace(dv)
        dv
      case sv: SparseVector =>
        throw new RuntimeException("Unexpected error in GCForestClassificationModel:" +
          " raw2probabilityInPlace encountered SparseVector")
    }
  }

  override def copy(extra: ParamMap): GCForestClassificationModel = {
    copyValues(new GCForestClassificationModel(uid,  scanModel, cascadeForest, numClasses), extra)
  }
}

class EnsembleRandomForestModel private[ml] (val forests: Array[RandomForestClassificationModel])

class MultiGrainedScanModel(val width: Int,
                       val height: Int,
                       val rfModel: RandomForestClassificationModel,
                       val crtfModel: RandomForestClassificationModel
                      ) {}