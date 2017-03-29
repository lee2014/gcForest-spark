package org.apache.spark.ml.classification

import org.apache.hadoop.fs.Path
import org.json4s.{DefaultFormats, JObject}
import org.json4s.JsonDSL._
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param._
import org.apache.spark.ml.tree.GCForestParams
import org.apache.spark.ml.util._
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

  /**
    * Scan image-style data
    * @param dataset raw input features
    * @param windowWidth the width of window
    * @param windowHeight the height of window
    * @return
    */
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

    val windowInstances = dataset.withColumn($(featuresCol), vector2Matrix(col($(featuresCol)), lit(width), lit(height)))
      .rdd.zipWithIndex.flatMap {
        case (row, index) =>
          val rows = Array.fill[Row]((width - windowWidth + 1) * (height - windowHeight + 1))(null)
          val featureIdx = row.fieldIndex($(featuresCol))
          val matrix = row.getAs[DenseMatrix]($(featuresCol))
          val features = Array.fill[Double](windowWidth * windowHeight)(0d)
          val newRow = ArrayBuffer[Any]() ++= row.toSeq

          Range(0, width - windowWidth + 1).foreach { x_offset =>
            Range(0, height - windowHeight + 1).foreach { y_offset =>
              Range(0, windowWidth).foreach { x =>
                Range(0, windowHeight).foreach { y =>
                  features(x * windowWidth + y) = matrix(x + x_offset, y + y_offset)
                }
              }
              val windowNum = x_offset * (width - windowWidth + 1) + y_offset
              newRow(featureIdx) = new DenseVector(features)
              newRow += windowNum.toLong
              newRow += index
              rows(windowNum) = Row.fromSeq(newRow)
            }
          }
          rows
    }

    val newSchema = schema
      .add(StructField($(windowCol), LongType))
      .add(StructField($(instanceCol), LongType))

    sparkSession.createDataFrame(windowInstances, newSchema)
  }

  /**
    * Use cross-validation to build k classes distribution features
    * @param dataset window features
    * @param rfc random forest classifier
    * @return k classes distribution features and random forest model
    */
  def featureTransform(dataset: Dataset[_], rfc: RandomTreeForest): (DataFrame, RandomTreeForestModel) = {
    val schema = dataset.schema
    val sparkSession = dataset.sparkSession
    var out: DataFrame = null

    // cross-validation for k classes distribution features
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

  // create a random forest classifier by type
  def genRFClassifier(rfType: String,
                          treeNum: Int,
                          minInstancePerNode: Int
                         ): RandomTreeForest = {
    val rf = rfType match {
      case "rf" => new RandomTreeForest()
      case "crtf" => new CompleteRandomTreeForestClassifier()
    }

    rf.setNumTrees(treeNum)
      .setMaxBins(Int.MaxValue)
      .setMaxDepth(30)
      .setMinInstancesPerNode(minInstancePerNode)
      .setFeatureSubsetStrategy("sqrt")
  }

  /**
    * Concate multi-scan features
    * @param dataset one of a window
    * @param sets the others
    * @return input for Cascade Forest
    */
  def concatenate(dataset: Dataset[_], sets: Dataset[_]*): DataFrame = {
    val sparkSession = dataset.sparkSession
    var unionSet = dataset.toDF
    sets.foreach(ds => unionSet = unionSet.union(ds.toDF))

    class Record(val instance: Long,    // instance id
                 val label: Double,     // label
                 val features: Vector,  // features
                 val scanId: Int,       // the scan id for multi-scan
                 val treeNum: Int,      // tree id
                 val winId: Long)       // window id

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

  /*
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
  */

  /**
    * concat inputs of Cascade Forest with prediction
    * @param feature input features
    * @param predict prediction features
    * @return
    */
  def mergeFeatureAndPredict(feature: Dataset[_], predict: Dataset[_]): DataFrame = {
    val vectorMerge = udf { (v1: Vector, v2: Vector) =>
      new DenseVector(v1.toArray.union(v2.toArray))
    }

    if (predict != null) {
      feature.join(
        predict.withColumnRenamed($(featuresCol), $(predictionCol)),
        Seq($(instanceCol))
      ).withColumn(
        $(featuresCol), vectorMerge(col($(featuresCol)), col($(predictionCol)))
      ).select($(instanceCol), $(featuresCol), $(labelCol)).toDF
    } else {
      feature.toDF
    }
  }

  override protected def train(dataset: Dataset[_]): GCForestClassificationModel = {
    val numClasses: Int = getNumClasses(dataset)
    var scanFeature: DataFrame = null
    val mgsModels = ArrayBuffer[MultiGrainedScanModel]()
    val erfModels = ArrayBuffer[Array[RandomTreeForestModel]]()

    /**
      *  Multi-Grained Scanning
      */
    if ($(dataStyle) == "image") {
      require($(multiScanWindow).length % 2 == 0,
        "The multiScanWindow must has the even number for image-style data")

      val scanFeatures = ArrayBuffer[Dataset[_]]()

      Range(0, $(multiScanWindow).length / 2).foreach { i =>
        // Get the size of scan window
        val (w, h) = ($(multiScanWindow)(i), $(multiScanWindow)(i+1))
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

    // Init classifiers
    Range(0, $(cascadeForestMaxIteration)).foreach { it =>
      val ensembleRandomForest = Array[RandomTreeForest](
        genRFClassifier("rf", $(cascadeForestTreeNum), $(cascadeForestMinInstancesPerNode)),
        genRFClassifier("rf", $(cascadeForestTreeNum), $(cascadeForestMinInstancesPerNode)),
        genRFClassifier("crtf", $(cascadeForestTreeNum), $(cascadeForestMinInstancesPerNode)),
        genRFClassifier("crtf", $(cascadeForestTreeNum), $(cascadeForestMinInstancesPerNode))
      )

      val training = mergeFeatureAndPredict(scanFeature, lastPrediction)
      var Array(growingSet, estimatingSet) = training.randomSplit(Array(0.8d, 0.2d), $(seed))

      val models = ensembleRandomForest.map(classifier => classifier.fit(growingSet))
      erfModels += models

      if (it < $(cascadeForestMaxIteration) - 1) {
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
    }

    scanFeature.unpersist
    new GCForestClassificationModel(mgsModels.toArray, erfModels.toArray, numClasses)
  }

  override def copy(extra: ParamMap): GCForestClassifier = defaultCopy(extra)
}


class GCForestClassificationModel private[ml] (override val uid: String,
                                               private val scanModel: Array[MultiGrainedScanModel],
                                               private val cascadeForest: Array[Array[RandomTreeForestModel]],
                                               override val numClasses: Int)
  extends ProbabilisticClassificationModel[Vector, GCForestClassificationModel]
    with GCForestParams with MLWritable with Serializable {

  def this(
            scanModel: Array[MultiGrainedScanModel],
            cascadeForest: Array[Array[RandomTreeForestModel]],
            numClasses: Int) =
    this(Identifiable.randomUID("gcfc"), scanModel, cascadeForest, numClasses)

  val numScans: Int = scanModel.length
  val numCascades: Int = cascadeForest.length

  def predictScanFeature(features: Vector): Vector = {
    val avgPredict = Array.fill[Double](numClasses)(0d)
    var lastPredict = Array[Double]()

    cascadeForest.foreach { models =>
      lastPredict = models.flatMap(
        m => m.predictProbability(new DenseVector(features.toArray.union(lastPredict))).toArray
      )
    }

    lastPredict.indices.foreach { i =>
      val classType = i % numClasses
      avgPredict(classType) = avgPredict(classType) + lastPredict(i)
    }

    new DenseVector(avgPredict)
  }

  override def predictRaw(features: Vector): Vector = {
    val scanFeatures = ArrayBuffer[Double]()

    /**
      * Multi-Grained Scanning
      */
    if ($(dataStyle) == "image") {
      val width = $(dataSize)(0)
      val height = $(dataSize)(1)
      require(features.size == width * height)

      val matrix = new DenseMatrix(width, height, features.toArray)

      scanModel.foreach { model =>
        val windowWidth = model.windowWidth
        val windowHeight = model.windowHeight
        val windowFeatures = Array.fill[Double](windowWidth * windowHeight)(0)

        Seq(model.rfModel, model.crtfModel).foreach { featureModel =>
          Range(0, width - windowWidth + 1).foreach { x_offset =>
            Range(0, height - windowHeight + 1).foreach { y_offset =>
              Range(0, windowWidth).foreach { x =>
                Range(0, windowHeight).foreach { y =>
                  windowFeatures(x * windowWidth + y) = matrix(x + x_offset, y + y_offset)
                }
              }
              scanFeatures ++= featureModel.predictProbability(new DenseVector(windowFeatures)).toArray
            }
          }
        }
      }
    } else if ($(dataStyle) == "sequence") { // TODO
      throw new UnsupportedOperationException("Unsupported sequence data rightly!")
    } else {
      throw new UnsupportedOperationException("The dataStyle : " + $(dataStyle) + " is unsupported!")
    }

    /**
      *  Cascade Predicting
      */
    predictScanFeature(new DenseVector(scanFeatures.toArray))
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        ProbabilisticClassificationModel.normalizeToProbabilitiesInPlace(dv)
        dv
      case _: SparseVector =>
        throw new RuntimeException("Unexpected error in GCForestClassificationModel:" +
          " raw2probabilityInPlace encountered SparseVector")
    }
  }

  override protected def transformImpl(dataset: Dataset[_]): DataFrame = {
    val bcastModel = dataset.sparkSession.sparkContext.broadcast(this)
    val predictUDF = udf { (features: Any) =>
      bcastModel.value.predict(features.asInstanceOf[Vector])
    }
    dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
  }

  override def copy(extra: ParamMap): GCForestClassificationModel = {
    copyValues(new GCForestClassificationModel(uid,  scanModel, cascadeForest, numClasses), extra)
  }

  override def write: MLWriter =
    new GCForestClassificationModel.GCForestClassificationModelWriter(this)
}

/**
  * The metadata of GCForestClassificationModel
  *
  * root                  // the root directory of GCForestClassificationModel
  *  |--metadata          // metadata of GCForestClassificationModel
  *  |--scan              // Multi-Grained Scanning
  *  |   |--0
  *  |   |  |--metadata
  *  |   |  |--rf
  *  |   |  |--crtf
  *  |   |--1
  *  |   |--2
  *  |--cascade           // Cascade Forest
  *  |   |--0             // the level of Cascade Forest
  *  |   |  |--0          // the number of Forest
  *  |   |  |--1
  *  |   |  |--2
  *  |   |  |--3
  *  |   |--1
  *  |   |--2
  */
object GCForestClassificationModel extends MLReadable[GCForestClassificationModel] {
  override def read: MLReader[GCForestClassificationModel] = new GCForestClassificationModelReader

  override def load(path: String): GCForestClassificationModel = super.load(path)

  private[GCForestClassificationModel]
  class GCForestClassificationModelWriter(instance: GCForestClassificationModel)
    extends MLWriter {

    override protected def saveImpl(path: String): Unit = {

      val gcMetadata: JObject = Map(
        "numClasses" -> instance.numClasses,
        "numScans" -> instance.numScans,
        "numCascades" -> instance.numCascades)
      DefaultParamsWriter.saveMetadata(instance, path, sparkSession.sparkContext, Some(gcMetadata))

      // scanModel
      val scanPath = new Path(path, "scan").toString
      instance.scanModel.zipWithIndex.foreach { case (model, index) =>
        val modelPath = new Path(scanPath, index.toString).toString
        val metadata: JObject = Map(
          "windowWidth" -> model.windowWidth,
          "windowHeight" -> model.windowHeight
        )
        DefaultParamsWriter.saveMetadata(model, modelPath, sparkSession.sparkContext, Some(metadata))
        val rfModelPath = new Path(modelPath, "rf").toString
        model.rfModel.save(rfModelPath)
        val crtfModelPath = new Path(modelPath, "crtf").toString
        model.crtfModel.save(crtfModelPath)
      }

      // CascadeForestModel
      val cascadePath = new Path(path, "cascade").toString
      instance.cascadeForest.zipWithIndex.foreach { case(models, level) =>
        val modelsPath = new Path(cascadePath, level.toString).toString
        models.zipWithIndex.foreach { case(model, index) =>
          val modelPath = new Path(modelsPath, index.toString).toString
          model.save(modelPath)
        }
      }
    }
  }

  private class GCForestClassificationModelReader
    extends MLReader[GCForestClassificationModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[GCForestClassificationModel].getName
    val mgsClassName: String = classOf[MultiGrainedScanModel].getName

    override def load(path: String): GCForestClassificationModel = {
      implicit val format = DefaultFormats
      val gcMetadata = DefaultParamsReader.loadMetadata(path, sparkSession.sparkContext, className)

      val numClasses = (gcMetadata.metadata \ "numClasses").extract[Int]
      val numScans = (gcMetadata.metadata \ "numScans").extract[Int]
      val numCascades = (gcMetadata.metadata \ "numCascades").extract[Int]

      val scanPath = new Path(path, "scan").toString
      val scanModel = Range(0, numScans).map { index =>
        val modelPath = new Path(scanPath, index.toString).toString
        val scanMetadata = DefaultParamsReader.loadMetadata(path, sparkSession.sparkContext, mgsClassName)
        val windowWidth = (scanMetadata.metadata \ "windowWidth").extract[Int]
        val windowHeight = (scanMetadata.metadata \ "windowHeight").extract[Int]
        val rfPath = new Path(modelPath, "rf").toString
        val rfModel = RandomTreeForestModel.load(rfPath)
        val crtfPath = new Path(modelPath, "crtf").toString
        val crtfModel = RandomTreeForestModel.load(crtfPath)
        new MultiGrainedScanModel(windowWidth, windowHeight, rfModel, crtfModel)
      }.toArray

      val cascadePath = new Path(path, "cascade").toString
      val cascadeForest = Range(0, numCascades).map { level =>
        val modelsPath = new Path(cascadePath, level.toString).toString
        Range(0, 4).map { index =>
          val modelPath = new Path(modelsPath, index.toString).toString
          RandomTreeForestModel.load(modelPath)
        }.toArray
      }.toArray

      val gcForestModel = new GCForestClassificationModel(gcMetadata.uid, scanModel, cascadeForest, numClasses)

      DefaultParamsReader.getAndSetParams(gcForestModel, gcMetadata)
      gcForestModel
    }
  }
}

class MultiGrainedScanModel(override val uid: String,
                            val windowWidth: Int, val windowHeight: Int,
                            val rfModel: RandomTreeForestModel,
                            val crtfModel: RandomTreeForestModel) extends Params {
  def this(
            windowWidth: Int, windowHeight: Int,
            rfModel: RandomTreeForestModel,
            crtfModel: RandomTreeForestModel) =
    this(Identifiable.randomUID("mgs"), windowWidth, windowHeight, rfModel, crtfModel)

  override def copy(extra: ParamMap): Params =
    copyValues(new MultiGrainedScanModel(uid, windowWidth, windowHeight, rfModel, crtfModel), extra)
}